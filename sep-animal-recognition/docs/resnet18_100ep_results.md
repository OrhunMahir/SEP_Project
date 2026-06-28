# ResNet-18 100 Epoch Results

This document summarizes the 100-epoch scratch ResNet-18 run before threshold calibration.

## Run Summary

| Metric | Value |
|---|---:|
| Experiment | `resnet18_yolo_crop_padded_100ep` |
| Model | ResNet-18 scratch |
| Input pipeline | YOLO crop + square padding + raw fallback |
| Device | `cuda` |
| Trainable parameters | 11,187,285 |
| Epochs completed | 100 |
| Elapsed time | 19.84 min |
| Best validation macro-F1 | 0.7772 |

## Final Epoch Metrics

These are the metrics printed at epoch `100/100`.

| Metric | Value |
|---|---:|
| Train loss | 0.6518 |
| Validation loss | 1.2516 |
| Accuracy | 0.7594 |
| Macro precision | 0.7598 |
| Macro recall | 0.7784 |
| Macro-F1 | 0.7668 |
| Weighted-F1 | 0.7581 |
| Reject precision | 0.7643 |
| Reject recall | 0.7094 |
| Reject-F1 | 0.7358 |
| False accepts | 93 |
| False rejects | 70 |
| Learning rate | 0.000000 |

## Best Checkpoint Metrics

These are the best validation metrics saved in `summary.json`.

| Metric | Value |
|---|---:|
| Accuracy | 0.7714 |
| Macro precision | 0.7739 |
| Macro recall | 0.7855 |
| Macro-F1 | 0.7772 |
| Weighted precision | 0.7730 |
| Weighted recall | 0.7714 |
| Weighted-F1 | 0.7702 |
| Reject precision | 0.7680 |
| Reject recall | 0.7344 |
| Reject-F1 | 0.7508 |
| Reject support | 320 |
| False accepts | 85 |
| False rejects | 71 |
| Validation loss | 1.2687 |

## Per-Class Metrics

These per-class values are computed from the best validation checkpoint before confidence-threshold calibration.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.7000 | 0.7000 | 0.7000 | 40 |
| Bengal | 0.7209 | 0.7750 | 0.7470 | 40 |
| Birman | 0.8222 | 0.9250 | 0.8706 | 40 |
| Bombay | 0.7647 | 0.9750 | 0.8571 | 40 |
| British Shorthair | 0.9375 | 0.7500 | 0.8333 | 40 |
| Maine Coon | 0.7105 | 0.6750 | 0.6923 | 40 |
| Ragdoll | 0.8250 | 0.8250 | 0.8250 | 40 |
| Sphynx | 0.8378 | 0.7750 | 0.8052 | 40 |
| Tabby | 0.5714 | 0.5000 | 0.5333 | 40 |
| Tiger Cat | 0.6216 | 0.5750 | 0.5974 | 40 |
| Beagle | 0.7561 | 0.7750 | 0.7654 | 40 |
| Pug | 0.9211 | 0.8750 | 0.8974 | 40 |
| Boxer | 0.8205 | 0.8000 | 0.8101 | 40 |
| Shiba Inu | 0.7674 | 0.8250 | 0.7952 | 40 |
| Samoyed | 0.7551 | 0.9250 | 0.8315 | 40 |
| Golden Retriever | 0.6944 | 0.8333 | 0.7576 | 30 |
| German Shepherd | 0.6765 | 0.7667 | 0.7188 | 30 |
| Siberian Husky | 0.8611 | 0.8158 | 0.8378 | 38 |
| Dalmatian | 0.8857 | 0.8378 | 0.8611 | 37 |
| Rottweiler | 0.8333 | 0.8333 | 0.8333 | 30 |
| reject | 0.7680 | 0.7344 | 0.7508 | 320 |

## Notes

- The best checkpoint is selected by validation macro-F1.
- These values are not threshold-calibrated yet.
- After threshold calibration finishes, use `per_class_metrics_best_threshold.csv` and `confusion_matrix_best_threshold.png` for the final calibrated table and figure.
