# ResNet-18 100 Epoch Threshold Results

This document summarizes the threshold-calibrated ResNet-18 100-epoch scratch run.

## Threshold Summary

| Metric | Value |
|---|---:|
| Model | ResNet-18 scratch |
| Selection metric | `accuracy` |
| Best threshold | 0.2200 |
| Validation samples | 1085 |
| Checkpoint epoch | 85 |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7770 |
| Macro precision | 0.7851 |
| Macro recall | 0.7854 |
| Macro-F1 | 0.7827 |
| Weighted precision | 0.7788 |
| Weighted recall | 0.7770 |
| Weighted-F1 | 0.7759 |
| Reject precision | 0.7586 |
| Reject recall | 0.7562 |
| Reject-F1 | 0.7574 |
| Reject support | 320 |
| False accepts | 78 |
| False rejects | 77 |

## Baseline Without Threshold

| Metric | Value |
|---|---:|
| Accuracy | 0.7714 |
| Macro-F1 | 0.7772 |
| Weighted-F1 | 0.7702 |
| Reject-F1 | 0.7508 |
| False accepts | 85 |
| False rejects | 71 |

## Per-Class Metrics

These per-class values are computed after applying the selected confidence threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.7179 | 0.7000 | 0.7089 | 40 |
| Bengal | 0.7209 | 0.7750 | 0.7470 | 40 |
| Birman | 0.8409 | 0.9250 | 0.8810 | 40 |
| Bombay | 0.7800 | 0.9750 | 0.8667 | 40 |
| British Shorthair | 0.9677 | 0.7500 | 0.8451 | 40 |
| Maine Coon | 0.7500 | 0.6750 | 0.7105 | 40 |
| Ragdoll | 0.8250 | 0.8250 | 0.8250 | 40 |
| Sphynx | 0.8378 | 0.7750 | 0.8052 | 40 |
| Tabby | 0.5714 | 0.5000 | 0.5333 | 40 |
| Tiger Cat | 0.6389 | 0.5750 | 0.6053 | 40 |
| Beagle | 0.7561 | 0.7750 | 0.7654 | 40 |
| Pug | 0.9211 | 0.8750 | 0.8974 | 40 |
| Boxer | 0.8158 | 0.7750 | 0.7949 | 40 |
| Shiba Inu | 0.8049 | 0.8250 | 0.8148 | 40 |
| Samoyed | 0.7551 | 0.9250 | 0.8315 | 40 |
| Golden Retriever | 0.7143 | 0.8333 | 0.7692 | 30 |
| German Shepherd | 0.6765 | 0.7667 | 0.7188 | 30 |
| Siberian Husky | 0.8611 | 0.8158 | 0.8378 | 38 |
| Dalmatian | 0.9118 | 0.8378 | 0.8732 | 37 |
| Rottweiler | 0.8621 | 0.8333 | 0.8475 | 30 |
| reject | 0.7586 | 0.7562 | 0.7574 | 320 |

## Notes

- This table is threshold-calibrated.
- The selected threshold is chosen on the fixed internal validation split.
- Use the matching `confusion_matrix_best_threshold.png` file as the report figure.
