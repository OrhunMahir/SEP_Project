# Custom CNN 100 Epoch Threshold Results

This document summarizes the threshold-calibrated Custom CNN 100-epoch scratch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | Custom CNN 8-conv scratch |
| Selection metric | `accuracy` |
| Best threshold | 0.3600 |
| Validation samples | 1085 |
| Checkpoint epoch | 84 |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7797 |
| Macro precision | 0.8057 |
| Macro recall | 0.7986 |
| Macro-F1 | 0.7975 |
| Weighted precision | 0.7889 |
| Weighted recall | 0.7797 |
| Weighted-F1 | 0.7811 |
| Reject precision | 0.7259 |
| Reject recall | 0.7281 |
| Reject-F1 | 0.7270 |
| Reject support | 320 |
| False accepts | 87 |
| False rejects | 88 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.7594 |
| Macro-F1 | 0.7832 |
| Weighted-F1 | 0.7560 |
| Reject-F1 | 0.6693 |
| False accepts | 150 |
| False rejects | 18 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.8571 | 0.9000 | 0.8780 | 40 |
| Bengal | 0.9429 | 0.8250 | 0.8800 | 40 |
| Birman | 0.8529 | 0.7250 | 0.7838 | 40 |
| Bombay | 0.9744 | 0.9500 | 0.9620 | 40 |
| British Shorthair | 1.0000 | 0.7750 | 0.8732 | 40 |
| Maine Coon | 0.6444 | 0.7250 | 0.6824 | 40 |
| Ragdoll | 0.7561 | 0.7750 | 0.7654 | 40 |
| Sphynx | 0.8205 | 0.8000 | 0.8101 | 40 |
| Tabby | 0.6250 | 0.6250 | 0.6250 | 40 |
| Tiger Cat | 0.9032 | 0.7000 | 0.7887 | 40 |
| Beagle | 0.8056 | 0.7250 | 0.7632 | 40 |
| Pug | 0.8571 | 0.9000 | 0.8780 | 40 |
| Boxer | 0.8462 | 0.8250 | 0.8354 | 40 |
| Shiba Inu | 0.8824 | 0.7500 | 0.8108 | 40 |
| Samoyed | 0.8125 | 0.9750 | 0.8864 | 40 |
| Golden Retriever | 0.6512 | 0.9333 | 0.7671 | 30 |
| German Shepherd | 0.5897 | 0.7667 | 0.6667 | 30 |
| Siberian Husky | 0.6905 | 0.7632 | 0.7250 | 38 |
| Dalmatian | 0.9394 | 0.8378 | 0.8857 | 37 |
| Rottweiler | 0.7419 | 0.7667 | 0.7541 | 30 |
| reject | 0.7259 | 0.7281 | 0.7270 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- Use the matching `confusion_matrix_best_threshold.png` file as the report figure.
