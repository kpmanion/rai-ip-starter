import argparse

import pandas as pd
from transformers import CLIPTokenizer

from common import MODEL_ID, load_strategies

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="prompts/dev_prompts.csv")
ap.add_argument("--strategies", default="prompts/strategies.json")
args = ap.parse_args()

tok = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
limit = tok.model_max_length
strategies = load_strategies(args.strategies)
for _, p in pd.read_csv(args.prompts).iterrows():
    for name, template in strategies.items():
        text = template.format(caption=p["caption"])
        n = len(tok(text).input_ids)
        seen = tok.decode(tok(text, truncation=True, max_length=limit).input_ids,
                          skip_special_tokens=True)
        print(f"{p['prompt_id']} / {name}: {n} tokens (limit {limit})")
        if n > limit:
            print("  model actually reads:", seen)
