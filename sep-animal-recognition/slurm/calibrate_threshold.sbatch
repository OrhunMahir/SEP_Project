#!/usr/bin/env bash
#SBATCH --job-name=threshold-calibration
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Bu kısa jobda mevcut checkpoint'i eğitmeden validation üzerinde tarıyorum.
cd "$HOME/projects/SEP_Project_code/sep-animal-recognition"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -u scripts/calibrate_threshold.py --device cuda --num-workers 0
