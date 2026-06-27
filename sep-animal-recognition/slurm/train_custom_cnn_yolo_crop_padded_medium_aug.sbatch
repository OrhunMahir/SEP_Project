#!/usr/bin/env bash
#SBATCH --job-name=custom-cnn-yolo-padded-medium
#SBATCH --partition=NvidiaAll
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/train_baseline.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug.json \
  --device cuda \
  --num-workers 4
