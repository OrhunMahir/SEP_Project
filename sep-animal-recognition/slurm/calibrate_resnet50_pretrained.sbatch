#!/usr/bin/env bash
#SBATCH --job-name=resnet50-threshold
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -u scripts/calibrate_resnet50_pretrained.py \
  --config configs/resnet50_pretrained.json \
  --device cuda \
  --num-workers 0
