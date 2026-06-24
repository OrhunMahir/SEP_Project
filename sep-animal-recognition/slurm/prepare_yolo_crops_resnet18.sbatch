#!/usr/bin/env bash
#SBATCH --job-name=yolo-crop-cache-resnet18
#SBATCH --partition=NvidiaAll
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/prepare_yolo_crops.py \
  --config configs/resnet18_yolo_crop.json \
  --detector yolov8n.pt \
  --detector-device 0 \
  --confidence 0.25 \
  --batch-size 16
