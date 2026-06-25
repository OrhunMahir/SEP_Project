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
config_path="${CONFIG_PATH:-configs/custom_cnn_baseline.json}"
selection_metric="${SELECTION_METRIC:-macro_f1}"

python -u scripts/calibrate_threshold.py \
  --config "$config_path" \
  --selection-metric "$selection_metric" \
  --device cuda \
  --num-workers 0
