#!/usr/bin/env bash
#SBATCH --job-name=threshold-resnet18-yolo-100ep
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -u scripts/calibrate_threshold.py \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --selection-metric accuracy \
  --device cuda \
  --num-workers 0
