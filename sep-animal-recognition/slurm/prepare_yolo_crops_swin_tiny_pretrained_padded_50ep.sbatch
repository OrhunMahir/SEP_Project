#!/usr/bin/env bash
#SBATCH --job-name=yolo-crop-cache-swin-tiny-pretrained-50ep
#SBATCH --partition=NvidiaAll
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the sep-animal-recognition directory.}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/prepare_yolo_crops.py \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --detector yolov8n.pt \
  --detector-device 0 \
  --confidence 0.25 \
  --batch-size 16 \
  --padding-fraction 0.10 \
  --square-crop
