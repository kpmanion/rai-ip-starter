#!/bin/bash
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C gpu2080
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out

source env.sh
nvidia-smi -L
python scripts/generate.py --seeds 10
python scripts/score.py
python scripts/plot_scores.py
python scripts/magnitude.py --seeds 3
