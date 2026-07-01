#!/usr/bin/env bash
#SBATCH --job-name=threshold-swin-tiny-pretrained-yolo-50ep
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -u scripts/calibrate_threshold.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --selection-metric accuracy \
  --device cuda \
  --num-workers 0
