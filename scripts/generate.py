import argparse
import os

import pandas as pd
import torch

from common import MODEL_ID, load_pipe, load_strategies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/dev_prompts.csv")
    ap.add_argument("--strategies", default="prompts/strategies.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None
    strategies = load_strategies(args.strategies, only)
    prompts = pd.read_csv(args.prompts)
    pipe = load_pipe(args.model)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for _, p in prompts.iterrows():
        for name, template in strategies.items():
            text = template.format(caption=p["caption"])
            folder = os.path.join(args.out, "images", p["prompt_id"], name)
            os.makedirs(folder, exist_ok=True)
            for seed in range(args.seeds):
                path = os.path.join(folder, f"seed{seed}.png")
                if not os.path.exists(path):
                    g = torch.Generator("cuda").manual_seed(seed)
                    img = pipe(text, num_inference_steps=args.steps,
                               guidance_scale=args.guidance, generator=g).images[0]
                    img.save(path)
                rows.append(dict(prompt_id=p["prompt_id"], strategy=name, seed=seed,
                                 prompt_text=text, path=path))
            print(f"done {p['prompt_id']} / {name}")

    meta_path = os.path.join(args.out, "metadata.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(meta_path):
        new = pd.concat([pd.read_csv(meta_path), new]).drop_duplicates(subset="path", keep="last")
    new.to_csv(meta_path, index=False)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
