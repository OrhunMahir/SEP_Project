#!/usr/bin/env bash
#SBATCH --job-name=efficientnet-b0-ft
#SBATCH --partition=NvidiaAll
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/train_baseline.py \
  --config configs/efficientnet_b0_finetune.json \
  --device cuda \
  --num-workers 4
