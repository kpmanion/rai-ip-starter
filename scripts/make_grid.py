import argparse
import os

import pandas as pd
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_id", required=True)
ap.add_argument("--cols", type=int, default=5)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--out", default="outputs")
args = ap.parse_args()

meta = pd.read_csv(os.path.join(args.out, "metadata.csv"))
meta = meta[meta["prompt_id"] == args.prompt_id]
strategies = list(dict.fromkeys(meta["strategy"]))
pad = 24
canvas = Image.new("RGB", (args.cols * args.size, len(strategies) * (args.size + pad)), "white")
draw = ImageDraw.Draw(canvas)
for r, st in enumerate(strategies):
    y = r * (args.size + pad)
    draw.text((4, y + 6), st, fill="black")
    rows = meta[meta["strategy"] == st].sort_values("seed").head(args.cols)
    for c, path in enumerate(rows["path"]):
        im = Image.open(path).convert("RGB").resize((args.size, args.size))
        canvas.paste(im, (c * args.size, y + pad))
dest = os.path.join(args.out, f"grid_{args.prompt_id}.png")
canvas.save(dest)
print("saved", dest)
