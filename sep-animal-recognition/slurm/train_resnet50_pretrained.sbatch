#!/usr/bin/env bash
#SBATCH --job-name=resnet50-pretrained
#SBATCH --partition=NvidiaAll
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python -u scripts/train_resnet50_pretrained.py \
  --config configs/resnet50_pretrained.json \
  --device cuda \
  --num-workers 4
