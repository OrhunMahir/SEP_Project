#!/usr/bin/env bash
#SBATCH --job-name=threshold-custom-cnn-yolo-padded
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -u scripts/calibrate_threshold.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug.json \
  --selection-metric macro_f1 \
  --device cuda \
  --num-workers 0
