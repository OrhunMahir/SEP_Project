#!/usr/bin/env bash
#SBATCH --job-name=swin-gradcam
#SBATCH --partition=NvidiaAll
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the sep-animal-recognition directory.}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/runs/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

config_path="${CONFIG_PATH:-configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json}"
checkpoint_path="${CHECKPOINT_PATH:-runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt}"
manifest_path="${MANIFEST_PATH:-splits/val_seed42.csv}"
dataset_root="${DATASET_ROOT:?Set DATASET_ROOT to the directory containing manifest images.}"
target_class="${TARGET_CLASS:-predicted}"
target_layer="${TARGET_LAYER:-stage3_last_norm1}"
confidence_threshold="${CONFIDENCE_THRESHOLD:-0.0}"
output_dir="${OUTPUT_DIR:-runs/swin_gradcam}"
yolo_device="${YOLO_DEVICE:-0}"

if [[ ! -f "$config_path" ]]; then
  echo "Config was not found: $config_path" >&2
  exit 1
fi
if [[ ! -f "$checkpoint_path" ]]; then
  echo "Checkpoint was not found: $checkpoint_path" >&2
  exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "Manifest was not found: $manifest_path" >&2
  exit 1
fi
if [[ ! -d "$dataset_root" ]]; then
  echo "Dataset root was not found: $dataset_root" >&2
  exit 1
fi

extra_args=()
if [[ "${NO_YOLO:-0}" == "1" ]]; then
  extra_args+=(--no-yolo)
fi
if [[ -n "${LIMIT:-}" ]]; then
  extra_args+=(--limit "$LIMIT")
fi

nvidia-smi
python -u scripts/gradcam_swin.py \
  --config "$config_path" \
  --checkpoint "$checkpoint_path" \
  --manifest "$manifest_path" \
  --dataset-root "$dataset_root" \
  --target-class "$target_class" \
  --target-layer "$target_layer" \
  --device cuda \
  --yolo-device "$yolo_device" \
  --confidence-threshold "$confidence_threshold" \
  --output-dir "$output_dir" \
  "${extra_args[@]}"
