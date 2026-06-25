#!/usr/bin/env bash
#SBATCH --job-name=swin-tiny
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/SEP_Project/sep-animal-recognition"
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src"

nvidia-smi
python scripts/train_baseline.py \
  --config configs/swin_tiny_scratch.json \
  --device cuda \
  --num-workers 4
