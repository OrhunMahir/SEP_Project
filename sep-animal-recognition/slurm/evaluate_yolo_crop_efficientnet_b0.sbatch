#!/usr/bin/env bash
#SBATCH --job-name=yolo-crop-efficientnet-b0
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/evaluate_yolo_crop.py \
  --config configs/efficientnet_b0_scratch.json \
  --checkpoint runs/efficientnet_b0_scratch/best.pt \
  --threshold 0.41 \
  --detector yolov8n.pt \
  --device cuda \
  --detector-device 0 \
  --confidence 0.25 \
  --batch-size 16 \
  --output-dir runs/efficientnet_b0_yolo_crop
