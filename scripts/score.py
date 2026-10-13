import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from common import CLIP_ID, THRESHOLD


def embed(paths, model, proc, dev, bs=16):
    chunks = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            f = model.get_image_features(**x)
        if not torch.is_tensor(f):
            f = f.pooler_output
        chunks.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(chunks)


def find_ref(refs_dir, pid):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(refs_dir, f"{pid}.{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--refs", default="references")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.out, "metadata.csv")).reset_index(drop=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    emb = embed(meta["path"].tolist(), model, proc, dev)

    meta["sim_to_ref"] = float("nan")
    for pid in meta["prompt_id"].unique():
        ref = find_ref(args.refs, pid)
        if ref is None:
            print(f"no reference image for {pid}, skipping reference similarity")
            continue
        r = embed([ref], model, proc, dev)
        idx = meta.index[meta["prompt_id"] == pid].to_numpy()
        meta.loc[idx, "sim_to_ref"] = (emb[idx] @ r.T).squeeze(1).numpy()
    meta["above"] = meta["sim_to_ref"] > args.threshold

    selfsim = []
    for (pid, st), g in meta.groupby(["prompt_id", "strategy"]):
        e = emb[g.index.to_numpy()]
        n = len(g)
        val = float(((e @ e.T).sum() - n) / (n * (n - 1))) if n > 1 else float("nan")
        selfsim.append(dict(prompt_id=pid, strategy=st, self_sim=val))
    ss = pd.DataFrame(selfsim).groupby("strategy")["self_sim"].mean().to_frame("mean_self_sim")

    scored = meta.dropna(subset=["sim_to_ref"])
    ref_stats = scored.groupby("strategy").agg(n=("sim_to_ref", "size"),
                                               mean_sim=("sim_to_ref", "mean"),
                                               pct_above=("above", "mean"))
    ref_stats["pct_above"] = ref_stats["pct_above"] * 100
    summary = ss.join(ref_stats, how="left")

    meta.to_csv(os.path.join(args.out, "scores.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print(summary.round(3))


if __name__ == "__main__":
    main()
