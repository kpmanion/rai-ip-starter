import argparse
import os

import pandas as pd
import torch

from common import MODEL_ID, load_pipe, load_strategies


def embed_text(pipe, text):
    ids = pipe.tokenizer(text, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        return pipe.text_encoder(ids)[0]


def first_step_magnitude(pipe, cond, uncond, seed, steps):
    size = pipe.unet.config.sample_size
    g = torch.Generator("cuda").manual_seed(seed)
    latents = torch.randn((1, pipe.unet.config.in_channels, size, size),
                          generator=g, device="cuda", dtype=torch.float16)
    pipe.scheduler.set_timesteps(steps, device="cuda")
    latents = latents * pipe.scheduler.init_noise_sigma
    t = pipe.scheduler.timesteps[0]
    x = pipe.scheduler.scale_model_input(latents, t)
    with torch.no_grad():
        pred_c = pipe.unet(x, t, encoder_hidden_states=cond).sample
        pred_u = pipe.unet(x, t, encoder_hidden_states=uncond).sample
    return (pred_c - pred_u).float().flatten().norm().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/dev_prompts.csv")
    ap.add_argument("--strategies", default="prompts/strategies.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="outputs/magnitude.csv")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None
    strategies = load_strategies(args.strategies, only)
    prompts = pd.read_csv(args.prompts)
    pipe = load_pipe(args.model)
    uncond = embed_text(pipe, "")

    rows = []
    for _, p in prompts.iterrows():
        for name, template in strategies.items():
            cond = embed_text(pipe, template.format(caption=p["caption"]))
            for seed in range(args.seeds):
                m = first_step_magnitude(pipe, cond, uncond, seed, args.steps)
                rows.append(dict(prompt_id=p["prompt_id"], strategy=name, seed=seed, magnitude=m))
            print(f"done {p['prompt_id']} / {name}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("strategy")["magnitude"].mean().round(3))


if __name__ == "__main__":
    main()
