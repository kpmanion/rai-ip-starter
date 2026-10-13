import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
ap.add_argument("--threshold", type=float, default=0.85)
args = ap.parse_args()

df = pd.read_csv(os.path.join(args.out, "scores.csv")).dropna(subset=["sim_to_ref"])
if df.empty:
    raise SystemExit("no reference similarities found; add images to references/ and rerun score.py")
fig, ax = plt.subplots(figsize=(7, 4))
for st, g in df.groupby("strategy"):
    ax.hist(g["sim_to_ref"], bins=20, alpha=0.5, label=st)
ax.axvline(args.threshold, color="black", linestyle="--")
ax.set_xlabel("CLIP cosine similarity to reference image")
ax.set_ylabel("number of images")
ax.legend()
fig.tight_layout()
dest = os.path.join(args.out, "sim_hist.png")
fig.savefig(dest, dpi=150)
print("saved", dest)
