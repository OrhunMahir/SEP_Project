#!/usr/bin/env bash
#SBATCH --job-name=cnn8-medaug-confusion
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"

python -u scripts/confusion_matrix_custom_cnn_8conv.py \
  --config configs/custom_cnn_8conv_medium_aug.json \
  --threshold 0.34 \
  --device cuda \
  --num-workers 0
