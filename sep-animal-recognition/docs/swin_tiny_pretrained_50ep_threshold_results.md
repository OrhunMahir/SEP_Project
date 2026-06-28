# Swin Tiny Pretrained 50 Epoch Threshold Results

This document summarizes the threshold-calibrated pretrained Swin Tiny 50-epoch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | Swin Tiny pretrained |
| Input pipeline | YOLO crop + square padding |
| Config | `configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json` |
| Selection metric | `accuracy` |
| Best threshold | 0.3900 |
| Validation samples | 1085 |
| Checkpoint epoch | 29 |
| Epochs completed | 41 |
| Trainable parameters | 27,535,503 |
| Device | `cuda` |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9475 |
| Macro precision | 0.9392 |
| Macro recall | 0.9686 |
| Macro-F1 | 0.9527 |
| Weighted precision | 0.9498 |
| Weighted recall | 0.9475 |
| Weighted-F1 | 0.9472 |
| Reject precision | 0.9759 |
| Reject recall | 0.8875 |
| Reject-F1 | 0.9296 |
| Reject support | 320 |
| False accepts | 36 |
| False rejects | 7 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.9447 |
| Macro precision | 0.9331 |
| Macro recall | 0.9682 |
| Macro-F1 | 0.9492 |
| Weighted precision | 0.9479 |
| Weighted recall | 0.9447 |
| Weighted-F1 | 0.9445 |
| Reject precision | 0.9860 |
| Reject recall | 0.8781 |
| Reject-F1 | 0.9289 |
| Reject support | 320 |
| False accepts | 39 |
| False rejects | 4 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.9302 | 1.0000 | 0.9639 | 40 |
| Bengal | 0.9091 | 1.0000 | 0.9524 | 40 |
| Birman | 0.9500 | 0.9500 | 0.9500 | 40 |
| Bombay | 0.9750 | 0.9750 | 0.9750 | 40 |
| British Shorthair | 0.9744 | 0.9500 | 0.9620 | 40 |
| Maine Coon | 0.9512 | 0.9750 | 0.9630 | 40 |
| Ragdoll | 0.9750 | 0.9750 | 0.9750 | 40 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 40 |
| Tabby | 1.0000 | 0.8750 | 0.9333 | 40 |
| Tiger Cat | 0.9500 | 0.9500 | 0.9500 | 40 |
| Beagle | 0.8837 | 0.9500 | 0.9157 | 40 |
| Pug | 0.9524 | 1.0000 | 0.9756 | 40 |
| Boxer | 0.9302 | 1.0000 | 0.9639 | 40 |
| Shiba Inu | 0.9091 | 1.0000 | 0.9524 | 40 |
| Samoyed | 0.8889 | 1.0000 | 0.9412 | 40 |
| Golden Retriever | 0.9062 | 0.9667 | 0.9355 | 30 |
| German Shepherd | 0.8529 | 0.9667 | 0.9062 | 30 |
| Siberian Husky | 0.9231 | 0.9474 | 0.9351 | 38 |
| Dalmatian | 0.9474 | 0.9730 | 0.9600 | 37 |
| Rottweiler | 0.9375 | 1.0000 | 0.9677 | 30 |
| reject | 0.9759 | 0.8875 | 0.9296 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- The confusion matrix source is `confusion_matrix_best_threshold.csv`.
