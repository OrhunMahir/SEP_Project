#!/usr/bin/env bash
#SBATCH --job-name=resnet18-yolo-padded-80ep
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/train_baseline.py \
  --config configs/resnet18_yolo_crop_padded_80ep.json \
  --device cuda \
  --num-workers 4
