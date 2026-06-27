#!/usr/bin/env bash
#SBATCH --job-name=custom-cnn-8conv-wide-medaug
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
python -u scripts/train_custom_cnn_8conv_wide.py \
  --config configs/custom_cnn_8conv_wide_medium_aug.json \
  --device cuda \
  --num-workers 4
