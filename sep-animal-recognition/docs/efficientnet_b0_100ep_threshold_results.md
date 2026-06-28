# EfficientNet-B0 100 Epoch Threshold Results

This document summarizes the threshold-calibrated EfficientNet-B0 100-epoch scratch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | EfficientNet-B0 scratch |
| Selection metric | `accuracy` |
| Best threshold | 0.5200 |
| Validation samples | 1085 |
| Checkpoint epoch | 94 |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7475 |
| Macro precision | 0.7970 |
| Macro recall | 0.7419 |
| Macro-F1 | 0.7631 |
| Weighted precision | 0.7584 |
| Weighted recall | 0.7475 |
| Weighted-F1 | 0.7469 |
| Reject precision | 0.6455 |
| Reject recall | 0.7625 |
| Reject-F1 | 0.6991 |
| Reject support | 320 |
| False accepts | 76 |
| False rejects | 134 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.7456 |
| Macro-F1 | 0.7557 |
| Weighted-F1 | 0.7435 |
| Reject-F1 | 0.7079 |
| False accepts | 114 |
| False rejects | 56 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.6585 | 0.6750 | 0.6667 | 40 |
| Bengal | 0.6531 | 0.8000 | 0.7191 | 40 |
| Birman | 0.8889 | 0.8000 | 0.8421 | 40 |
| Bombay | 0.8372 | 0.9000 | 0.8675 | 40 |
| British Shorthair | 0.9412 | 0.8000 | 0.8649 | 40 |
| Maine Coon | 0.8519 | 0.5750 | 0.6866 | 40 |
| Ragdoll | 0.8000 | 0.8000 | 0.8000 | 40 |
| Sphynx | 0.9355 | 0.7250 | 0.8169 | 40 |
| Tabby | 0.5806 | 0.4500 | 0.5070 | 40 |
| Tiger Cat | 0.6923 | 0.4500 | 0.5455 | 40 |
| Beagle | 0.7619 | 0.8000 | 0.7805 | 40 |
| Pug | 0.9000 | 0.9000 | 0.9000 | 40 |
| Boxer | 0.9118 | 0.7750 | 0.8378 | 40 |
| Shiba Inu | 0.8286 | 0.7250 | 0.7733 | 40 |
| Samoyed | 0.7955 | 0.8750 | 0.8333 | 40 |
| Golden Retriever | 0.7500 | 0.8000 | 0.7742 | 30 |
| German Shepherd | 0.8182 | 0.6000 | 0.6923 | 30 |
| Siberian Husky | 0.9032 | 0.7368 | 0.8116 | 38 |
| Dalmatian | 0.8421 | 0.8649 | 0.8533 | 37 |
| Rottweiler | 0.7419 | 0.7667 | 0.7541 | 30 |
| reject | 0.6455 | 0.7625 | 0.6991 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- Use the matching `confusion_matrix_best_threshold.png` file as the report figure.
