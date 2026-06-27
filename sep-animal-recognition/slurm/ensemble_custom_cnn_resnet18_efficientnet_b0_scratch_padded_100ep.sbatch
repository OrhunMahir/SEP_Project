#!/usr/bin/env bash
#SBATCH --job-name=ensemble-scratch-100ep
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
python -u scripts/evaluate_ensemble.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --config configs/efficientnet_b0_yolo_crop_padded_100ep.json \
  --device cuda \
  --num-workers 0 \
  --selection-metric accuracy \
  --weight-step 0.05 \
  --output-dir runs/ensemble_custom_cnn_resnet18_efficientnet_b0_scratch_padded_100ep
