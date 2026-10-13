import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

from common import CLIP_ID, MODEL_ID

StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
CLIPModel.from_pretrained(CLIP_ID)
CLIPProcessor.from_pretrained(CLIP_ID)
print("models cached")
