#!/usr/bin/env bash
#SBATCH --job-name=cnn8-medaug-lr0007-threshold
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"

python -u scripts/calibrate_custom_cnn_8conv.py \
  --config configs/custom_cnn_8conv_medium_aug_lr0007.json \
  --device cuda \
  --num-workers 0 \
  --threshold-start 0.0 \
  --threshold-end 0.95 \
  --threshold-step 0.01
