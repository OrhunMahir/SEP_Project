# EfficientNet-B0 100 Epoch Results

This document summarizes the 100-epoch scratch EfficientNet-B0 run before threshold calibration.

## Run Summary

| Metric | Value |
|---|---:|
| Experiment | `efficientnet_b0_yolo_crop_padded_100ep` |
| Model | EfficientNet-B0 scratch |
| Input pipeline | YOLO crop + square padding + raw fallback |
| Device | `cuda` |
| Trainable parameters | 4,034,449 |
| Epochs completed | 100 |
| Elapsed time | 22.35 min |
| Best validation macro-F1 | 0.7557 |

## Final Epoch Metrics

These are the metrics printed at epoch `100/100`.

| Metric | Value |
|---|---:|
| Train loss | 0.6783 |
| Validation loss | 1.3370 |
| Accuracy | 0.7419 |
| Macro precision | 0.7316 |
| Macro recall | 0.7866 |
| Macro-F1 | 0.7539 |
| Weighted-F1 | 0.7396 |
| Reject precision | 0.8041 |
| Reject recall | 0.6156 |
| Reject-F1 | 0.6973 |
| False accepts | 123 |
| False rejects | 48 |
| Learning rate | 0.000000 |

## Best Checkpoint Metrics

These are the best validation metrics saved in `summary.json`.

| Metric | Value |
|---|---:|
| Accuracy | 0.7456 |
| Macro precision | 0.7380 |
| Macro recall | 0.7813 |
| Macro-F1 | 0.7557 |
| Weighted precision | 0.7509 |
| Weighted recall | 0.7456 |
| Weighted-F1 | 0.7435 |
| Reject precision | 0.7863 |
| Reject recall | 0.6438 |
| Reject-F1 | 0.7079 |
| Reject support | 320 |
| False accepts | 114 |
| False rejects | 56 |
| Validation loss | 1.3345 |

## Per-Class Metrics

These per-class values are computed from the best validation checkpoint before confidence-threshold calibration.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.5800 | 0.7250 | 0.6444 | 40 |
| Bengal | 0.6415 | 0.8500 | 0.7312 | 40 |
| Birman | 0.8333 | 0.8750 | 0.8537 | 40 |
| Bombay | 0.7115 | 0.9250 | 0.8043 | 40 |
| British Shorthair | 0.8462 | 0.8250 | 0.8354 | 40 |
| Maine Coon | 0.7879 | 0.6500 | 0.7123 | 40 |
| Ragdoll | 0.7674 | 0.8250 | 0.7952 | 40 |
| Sphynx | 0.8824 | 0.7500 | 0.8108 | 40 |
| Tabby | 0.5122 | 0.5250 | 0.5185 | 40 |
| Tiger Cat | 0.5526 | 0.5250 | 0.5385 | 40 |
| Beagle | 0.7609 | 0.8750 | 0.8140 | 40 |
| Pug | 0.8372 | 0.9000 | 0.8675 | 40 |
| Boxer | 0.7778 | 0.8750 | 0.8235 | 40 |
| Shiba Inu | 0.7500 | 0.7500 | 0.7500 | 40 |
| Samoyed | 0.7660 | 0.9000 | 0.8276 | 40 |
| Golden Retriever | 0.6857 | 0.8000 | 0.7385 | 30 |
| German Shepherd | 0.6786 | 0.6333 | 0.6552 | 30 |
| Siberian Husky | 0.7692 | 0.7895 | 0.7792 | 38 |
| Dalmatian | 0.8000 | 0.8649 | 0.8312 | 37 |
| Rottweiler | 0.7714 | 0.9000 | 0.8308 | 30 |
| reject | 0.7863 | 0.6438 | 0.7079 | 320 |

## Notes

- The best checkpoint is selected by validation macro-F1.
- These values are not threshold-calibrated yet.
- After threshold calibration finishes, use `per_class_metrics_best_threshold.csv` and `confusion_matrix_best_threshold.png` for the final calibrated table and figure.
