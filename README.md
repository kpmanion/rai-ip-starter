# Stress-testing prompt-based IP safeguards in text-to-image diffusion models

Team: Kent Manion and Lindsey Wang
Course: Responsible AI Engineering (CSDS 447), Case Western Reserve University
Status: Demo 1 (setup, data, baselines)

## What this repo does

Text-to-image diffusion models sometimes reproduce near-copies of their training images. The paper
"Safer Prompts: Reducing IP Risk in Visual Generative AI" (Reissinger et al., 2025) reports that rewriting
the text prompt (task-instruction, negative, and chain-of-thought styles) lowers how similar the generated
images are to training data. This repo reproduces that pipeline at small scale on Stable Diffusion 2 and
adds a first model-level measurement (Wen et al., ICLR 2024). Later stages will attack the rewrites
adversarially and look inside the model when they fail.

Demo 1 scope: one working environment, a small prompt set, the four prompt strategies as baselines,
CLIP similarity scoring, and the model-level magnitude signal.

## Hardware and software

- Cluster: Markov teaching cluster (account csds447), GPU partition markov_gpu
- GPU: RTX 2080 Ti (11 GB) is enough for Stable Diffusion 2 in fp16; use gpu4090 if it runs out of memory
- Python 3 with a virtual environment; packages listed in requirements.txt
- Exact installed versions are recorded in requirements-lock.txt (created with pip freeze)

## Setup on Markov (from a fresh clone)

```
mkdir -p -m 700 /mnt/vstor/courses/csds447/$USER
cd /mnt/vstor/courses/csds447/$USER
git clone https://github.com/kpmanion/rai-ip-starter
cd rai-ip-starter
source env.sh
python3 -m venv $CS_ROOT/venvs/rai
source env.sh
pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements-lock.txt
python scripts/download_models.py
```

env.sh points the Hugging Face, torch, and pip caches at the course folder (home has only 25 GB)
and activates the virtual environment. Run `source env.sh` at the start of every session.

## Quick check (on a GPU node)

```
srun -A csds447 -p markov_gpu -C gpu2080 --gres=gpu:1 -c 4 --mem=16G -t 01:00:00 --pty bash
source env.sh
python scripts/smoke_test.py
exit
```

You should see the GPU name, seconds per image, and outputs/smoke_test.png.

## Data and models

- Model: stabilityai/stable-diffusion-2 (Hugging Face), 30 steps, guidance scale 7.5, as in the paper
- Similarity model: openai/clip-vit-large-patch14 (the paper does not name its CLIP variant; this is our choice)
- Prompts: prompts/dev_prompts.csv (prompt_id, caption)
- Prompt strategies: prompts/strategies.json (baseline, task_instruction, negative, chain_of_thought).
  Fill in the three TODO templates from Table 1 of the paper before running.
- Reference images: references/p01.jpeg, references/p02.jpeg,  references/p03.jpeg one image per prompt. These are public-domain artworks, so they are committed to the repo. Source URLs are listed in references/SOURCES.txt.
- Splits: nothing is trained. Prompts are divided into a dev set, an evaluation set, and (later) an adversarial set.

## Baselines

1. No prompt engineering, task instruction, negative prompting, chain-of-thought (Safer Prompts, Table 1)
2. CLIP cosine similarity to the reference image, with 0.85 as the high-risk threshold (as in the paper)
3. Model-level signal: size of the difference between the prompt-conditioned and unconditioned noise prediction
   at the first denoising step (our reading of Wen et al., ICLR 2024)

## Running

```
python scripts/token_check.py
python scripts/generate.py --seeds 5
python scripts/score.py
python scripts/plot_scores.py
python scripts/make_grid.py --prompt_id p01
python scripts/magnitude.py --seeds 3
```

For longer runs: `sbatch job.sh`, then `squeue -u $USER`. Run generate, score, and magnitude inside a GPU job.
token_check.py needs no GPU.

Outputs go to outputs/ (not committed): images, metadata.csv, scores.csv, summary.csv, magnitude.csv,
grid_<prompt_id>.png, sim_hist.png.

## Layout

- scripts/ : all code
- prompts/ : captions and prompt templates
- references/ : reference images (local only)
- env.sh, job.sh, requirements.txt : environment and cluster job files

## Known differences from the paper

- Reference set is a small hand-picked set, not nearest-neighbor retrieval over LAION-Aesthetics 12M
- CLIP variant is our choice
- Stable Diffusion 2 (768) predicts v rather than noise, so magnitudes are not directly comparable to Wen et al.'s Stable Diffusion v1 numbers
- Prompts are truncated at 77 CLIP tokens; token_check.py shows what the model actually reads

## Sources

- Reissinger et al., Safer Prompts: Reducing IP Risk in Visual Generative AI (2025), arXiv 2505.03338
- Wen et al., Detecting, Explaining, and Mitigating Memorization in Diffusion Models (ICLR 2024)
- He et al., Fantastic Copyrighted Beasts and How (Not) to Generate Them (ICLR 2025)
- Kim et al., Automatic Jailbreaking of the Text-to-Image Generative AI Systems (2024)
- Pham et al., Circumventing Concept Erasure Methods for Text-to-Image Generative Models (ICLR 2024)

