# Custom CNN 100 Epoch Results

This document summarizes the 100-epoch scratch Custom CNN run before threshold calibration.

## Run Summary

| Metric | Value |
|---|---:|
| Experiment | `custom_cnn_yolo_crop_padded_medium_aug_100ep` |
| Model | Custom CNN 8-conv scratch |
| Input pipeline | YOLO crop + square padding + raw fallback |
| Device | `cuda` |
| Trainable parameters | 1,179,573 |
| Epochs completed | 100 |
| Elapsed time | 23.96 min |
| Best validation macro-F1 | 0.7832 |

## Final Epoch Metrics

These are the metrics printed at epoch `100/100`.

| Metric | Value |
|---|---:|
| Train loss | 1.0072 |
| Validation loss | 1.3948 |
| Accuracy | 0.7539 |
| Macro precision | 0.7402 |
| Macro recall | 0.8357 |
| Macro-F1 | 0.7760 |
| Weighted-F1 | 0.7491 |
| Reject precision | 0.9222 |
| Reject recall | 0.5188 |
| Reject-F1 | 0.6640 |
| False accepts | 154 |
| False rejects | 14 |
| Learning rate | 0.000000 |

## Best Checkpoint Metrics

These are the best validation metrics saved in `summary.json`.

| Metric | Value |
|---|---:|
| Accuracy | 0.7594 |
| Macro precision | 0.7503 |
| Macro recall | 0.8395 |
| Macro-F1 | 0.7832 |
| Weighted precision | 0.7941 |
| Weighted recall | 0.7594 |
| Weighted-F1 | 0.7560 |
| Reject precision | 0.9043 |
| Reject recall | 0.5312 |
| Reject-F1 | 0.6693 |
| Reject support | 320 |
| False accepts | 150 |
| False rejects | 18 |
| Validation loss | 1.3897 |

## Per-Class Metrics

These per-class values are computed from the best validation checkpoint before confidence-threshold calibration.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.7222 | 0.9750 | 0.8298 | 40 |
| Bengal | 0.8947 | 0.8500 | 0.8718 | 40 |
| Birman | 0.7750 | 0.7750 | 0.7750 | 40 |
| Bombay | 0.8837 | 0.9500 | 0.9157 | 40 |
| British Shorthair | 0.9697 | 0.8000 | 0.8767 | 40 |
| Maine Coon | 0.6038 | 0.8000 | 0.6882 | 40 |
| Ragdoll | 0.7021 | 0.8250 | 0.7586 | 40 |
| Sphynx | 0.7347 | 0.9000 | 0.8090 | 40 |
| Tabby | 0.5517 | 0.8000 | 0.6531 | 40 |
| Tiger Cat | 0.8000 | 0.8000 | 0.8000 | 40 |
| Beagle | 0.7561 | 0.7750 | 0.7654 | 40 |
| Pug | 0.7660 | 0.9000 | 0.8276 | 40 |
| Boxer | 0.8372 | 0.9000 | 0.8675 | 40 |
| Shiba Inu | 0.8421 | 0.8000 | 0.8205 | 40 |
| Samoyed | 0.7843 | 1.0000 | 0.8791 | 40 |
| Golden Retriever | 0.5957 | 0.9333 | 0.7273 | 30 |
| German Shepherd | 0.4800 | 0.8000 | 0.6000 | 30 |
| Siberian Husky | 0.5962 | 0.8158 | 0.6889 | 38 |
| Dalmatian | 0.8205 | 0.8649 | 0.8421 | 37 |
| Rottweiler | 0.7353 | 0.8333 | 0.7812 | 30 |
| reject | 0.9043 | 0.5312 | 0.6693 | 320 |

## Notes

- The best checkpoint is selected by validation macro-F1.
- These values are not threshold-calibrated yet.
- After threshold calibration finishes, use `per_class_metrics_best_threshold.csv` and `confusion_matrix_best_threshold.png` for the final calibrated table and figure.
