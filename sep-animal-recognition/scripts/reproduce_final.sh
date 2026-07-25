#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-cuda}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-0}"

python scripts/verify_submission.py --require-data

python scripts/prepare_yolo_crops.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --detector yolov8n.pt \
  --detector-device "$DETECTOR_DEVICE" \
  --confidence 0.25 \
  --padding-fraction 0.10 \
  --square-crop

FINAL_CONFIGS=(
  configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json
  configs/resnet18_yolo_crop_padded_100ep.json
  configs/efficientnet_b0_yolo_crop_padded_100ep.json
  configs/resnet18_pretrained_yolo_crop_padded_50ep.json
  configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json
  configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json
)

for config in "${FINAL_CONFIGS[@]}"; do
  python scripts/train_baseline.py --config "$config" --device "$DEVICE"
done

python scripts/evaluate_ensemble.py \
  --config configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json \
  --config configs/resnet18_yolo_crop_padded_100ep.json \
  --config configs/efficientnet_b0_yolo_crop_padded_100ep.json \
  --weight-step 0.05 \
  --threshold-start 0.00 \
  --threshold-end 0.95 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/reproduced_scratch_100ep_ensemble \
  --device "$DEVICE"

python scripts/evaluate_ensemble.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --weight-step 0.05 \
  --threshold-start 0.00 \
  --threshold-end 0.95 \
  --threshold-step 0.01 \
  --selection-metric accuracy \
  --output-dir runs/reproduced_pretrained_50ep_ensemble \
  --device "$DEVICE"

if [[ -n "${OFFICIAL_TEST_ROOT:-}" ]]; then
  python scripts/evaluate_official_ensemble.py \
    --image-folder "$OFFICIAL_TEST_ROOT" \
    --preset scratch_100ep \
    --output-dir runs/reproduced_official_scratch_100ep \
    --device "$DEVICE" \
    --detector-device "$DETECTOR_DEVICE"

  python scripts/evaluate_official_ensemble.py \
    --image-folder "$OFFICIAL_TEST_ROOT" \
    --preset pretrained_50ep \
    --output-dir runs/reproduced_official_pretrained_50ep \
    --device "$DEVICE" \
    --detector-device "$DETECTOR_DEVICE"
fi

python scripts/verify_submission.py --require-data --require-checkpoints
