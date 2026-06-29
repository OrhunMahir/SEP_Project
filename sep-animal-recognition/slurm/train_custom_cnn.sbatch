#!/usr/bin/env bash
#SBATCH --job-name=custom-cnn
#SBATCH --partition=NvidiaAll
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Submit the training job to Slurm without changing local project files.
cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

nvidia-smi
python scripts/train_baseline.py --device cuda --num-workers 4
