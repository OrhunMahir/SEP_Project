#!/usr/bin/env bash
#SBATCH --job-name=ensemble-resnet18-efficientnet-b0-padded
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python -u scripts/evaluate_ensemble.py \
  --config configs/resnet18_yolo_crop_padded.json \
  --config configs/efficientnet_b0_yolo_crop_padded.json \
  --device cuda \
  --num-workers 0 \
  --selection-metric accuracy \
  --model-0-weights 0.50,0.60,0.70,0.80,0.90 \
  --output-dir runs/ensemble_resnet18_efficientnet_b0_padded
