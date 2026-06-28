# EfficientNet-B0 Pretrained 50 Epoch Threshold Results

This document summarizes the threshold-calibrated pretrained EfficientNet-B0 50-epoch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | EfficientNet-B0 pretrained |
| Input pipeline | YOLO crop + square padding |
| Config | `configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json` |
| Selection metric | `accuracy` |
| Best threshold | 0.3700 |
| Validation samples | 1085 |
| Checkpoint epoch | 30 |
| Epochs completed | 42 |
| Trainable parameters | 4,034,449 |
| Device | `cuda` |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9410 |
| Macro precision | 0.9349 |
| Macro recall | 0.9526 |
| Macro-F1 | 0.9429 |
| Weighted precision | 0.9421 |
| Weighted recall | 0.9410 |
| Weighted-F1 | 0.9407 |
| Reject precision | 0.9604 |
| Reject recall | 0.9094 |
| Reject-F1 | 0.9342 |
| Reject support | 320 |
| False accepts | 29 |
| False rejects | 12 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.9382 |
| Macro precision | 0.9254 |
| Macro recall | 0.9544 |
| Macro-F1 | 0.9387 |
| Weighted precision | 0.9405 |
| Weighted recall | 0.9382 |
| Weighted-F1 | 0.9380 |
| Reject precision | 0.9795 |
| Reject recall | 0.8938 |
| Reject-F1 | 0.9346 |
| Reject support | 320 |
| False accepts | 34 |
| False rejects | 6 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.9750 | 0.9750 | 0.9750 | 40 |
| Bengal | 0.9048 | 0.9500 | 0.9268 | 40 |
| Birman | 0.9091 | 1.0000 | 0.9524 | 40 |
| Bombay | 0.9500 | 0.9500 | 0.9500 | 40 |
| British Shorthair | 0.9722 | 0.8750 | 0.9211 | 40 |
| Maine Coon | 0.9474 | 0.9000 | 0.9231 | 40 |
| Ragdoll | 0.9231 | 0.9000 | 0.9114 | 40 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 40 |
| Tabby | 0.9024 | 0.9250 | 0.9136 | 40 |
| Tiger Cat | 0.9189 | 0.8500 | 0.8831 | 40 |
| Beagle | 0.9091 | 1.0000 | 0.9524 | 40 |
| Pug | 0.9512 | 0.9750 | 0.9630 | 40 |
| Boxer | 0.9524 | 1.0000 | 0.9756 | 40 |
| Shiba Inu | 0.9750 | 0.9750 | 0.9750 | 40 |
| Samoyed | 0.8864 | 0.9750 | 0.9286 | 40 |
| Golden Retriever | 0.9062 | 0.9667 | 0.9355 | 30 |
| German Shepherd | 0.9333 | 0.9333 | 0.9333 | 30 |
| Siberian Husky | 0.9268 | 1.0000 | 0.9620 | 38 |
| Dalmatian | 0.9211 | 0.9459 | 0.9333 | 37 |
| Rottweiler | 0.9091 | 1.0000 | 0.9524 | 30 |
| reject | 0.9604 | 0.9094 | 0.9342 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- The confusion matrix source is `confusion_matrix_best_threshold.csv`.
