# ResNet-18 Pretrained 50 Epoch Threshold Results

This document summarizes the threshold-calibrated pretrained ResNet-18 50-epoch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | ResNet-18 pretrained |
| Input pipeline | YOLO crop + square padding |
| Config | `configs/resnet18_pretrained_yolo_crop_padded_50ep.json` |
| Selection metric | `accuracy` |
| Best threshold | 0.1600 |
| Validation samples | 1085 |
| Checkpoint epoch | 47 |
| Epochs completed | 50 |
| Trainable parameters | 11,187,285 |
| Device | `cuda` |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9364 |
| Macro precision | 0.9387 |
| Macro recall | 0.9476 |
| Macro-F1 | 0.9423 |
| Weighted precision | 0.9375 |
| Weighted recall | 0.9364 |
| Weighted-F1 | 0.9363 |
| Reject precision | 0.9323 |
| Reject recall | 0.9031 |
| Reject-F1 | 0.9175 |
| Reject support | 320 |
| False accepts | 31 |
| False rejects | 21 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.9355 |
| Macro precision | 0.9374 |
| Macro recall | 0.9474 |
| Macro-F1 | 0.9416 |
| Weighted precision | 0.9365 |
| Weighted recall | 0.9355 |
| Weighted-F1 | 0.9354 |
| Reject precision | 0.9320 |
| Reject recall | 0.9000 |
| Reject-F1 | 0.9157 |
| Reject support | 320 |
| False accepts | 32 |
| False rejects | 21 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.9512 | 0.9750 | 0.9630 | 40 |
| Bengal | 0.9268 | 0.9500 | 0.9383 | 40 |
| Birman | 1.0000 | 1.0000 | 1.0000 | 40 |
| Bombay | 0.9091 | 1.0000 | 0.9524 | 40 |
| British Shorthair | 0.9730 | 0.9000 | 0.9351 | 40 |
| Maine Coon | 0.9500 | 0.9500 | 0.9500 | 40 |
| Ragdoll | 0.9756 | 1.0000 | 0.9877 | 40 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 40 |
| Tabby | 1.0000 | 0.9000 | 0.9474 | 40 |
| Tiger Cat | 0.8974 | 0.8750 | 0.8861 | 40 |
| Beagle | 0.9487 | 0.9250 | 0.9367 | 40 |
| Pug | 0.9512 | 0.9750 | 0.9630 | 40 |
| Boxer | 0.9302 | 1.0000 | 0.9639 | 40 |
| Shiba Inu | 0.9268 | 0.9500 | 0.9383 | 40 |
| Samoyed | 0.8478 | 0.9750 | 0.9070 | 40 |
| Golden Retriever | 0.9000 | 0.9000 | 0.9000 | 30 |
| German Shepherd | 0.9655 | 0.9333 | 0.9492 | 30 |
| Siberian Husky | 0.8462 | 0.8684 | 0.8571 | 38 |
| Dalmatian | 0.9714 | 0.9189 | 0.9444 | 37 |
| Rottweiler | 0.9091 | 1.0000 | 0.9524 | 30 |
| reject | 0.9323 | 0.9031 | 0.9175 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- The confusion matrix source is `confusion_matrix_best_threshold.csv`.
