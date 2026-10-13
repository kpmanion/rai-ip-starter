import os
import time

import torch

from common import load_pipe

print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

pipe = load_pipe()
os.makedirs("outputs", exist_ok=True)
g = torch.Generator("cuda").manual_seed(0)
t0 = time.time()
img = pipe("a watercolor painting of a lighthouse", num_inference_steps=30,
           guidance_scale=7.5, generator=g).images[0]
print(f"seconds for one image: {time.time() - t0:.1f}")
img.save("outputs/smoke_test.png")
print("saved outputs/smoke_test.png")
