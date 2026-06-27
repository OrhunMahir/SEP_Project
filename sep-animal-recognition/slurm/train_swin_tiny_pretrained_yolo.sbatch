#!/usr/bin/env bash
#SBATCH --job-name=swin-tiny-yolo
#SBATCH --partition=Gobi,Luna
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/SEP_Project/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/train_baseline.py \
  --config configs/swin_tiny_pretrained_yolo.json \
  --device cuda \
  --num-workers 4
