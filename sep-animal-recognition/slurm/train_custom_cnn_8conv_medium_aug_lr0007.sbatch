#!/usr/bin/env bash
#SBATCH --job-name=custom-cnn-8conv-medaug-lr0007
#SBATCH --partition=NvidiaAll
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"

nvidia-smi
python -u scripts/train_custom_cnn_8conv_strong_aug.py \
  --config configs/custom_cnn_8conv_medium_aug_lr0007.json \
  --device cuda \
  --num-workers 4
