import json

import torch

MODEL_ID = "stabilityai/stable-diffusion-2"
CLIP_ID = "openai/clip-vit-large-patch14"
THRESHOLD = 0.85


def load_strategies(path, only=None):
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for name, template in raw.items():
        if only and name not in only:
            continue
        if template.startswith("TODO"):
            print(f"skipping '{name}': template not filled in")
            continue
        out[name] = template
    return out


def load_pipe(model_id=MODEL_ID):
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Run this inside an srun or sbatch GPU job.")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe
