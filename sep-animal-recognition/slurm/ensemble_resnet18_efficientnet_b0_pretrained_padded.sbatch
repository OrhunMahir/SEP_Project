#!/usr/bin/env bash
#SBATCH --job-name=ensemble-resnet18-efficientnet-b0-pretrained-padded
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
  --config configs/resnet18_pretrained_yolo_crop_padded.json \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded.json \
  --device cuda \
  --num-workers 0 \
  --selection-metric accuracy \
  --model-0-weights 0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00 \
  --output-dir runs/ensemble_resnet18_efficientnet_b0_pretrained_padded
