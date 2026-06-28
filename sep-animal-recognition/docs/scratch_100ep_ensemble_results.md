# Scratch 100 Epoch Ensemble Results

This document summarizes the threshold-calibrated scratch ensemble that combines the 100-epoch Custom CNN, ResNet-18, and EfficientNet-B0 runs.

## Ensemble Summary

| Metric | Value |
|---|---:|
| Selection metric | `accuracy` |
| Best threshold | 0.3200 |
| Validation samples | 1085 |
| Device | `cuda` |

## Ensemble Members

| Index | Model | Weight | Checkpoint epoch | Config |
|---:|---|---:|---:|---|
| 0 | Custom CNN 8-conv scratch | 0.4500 | 84 | `configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json` |
| 1 | ResNet-18 scratch | 0.2500 | 85 | `configs/resnet18_yolo_crop_padded_100ep.json` |
| 2 | EfficientNet-B0 scratch | 0.3000 | 94 | `configs/efficientnet_b0_yolo_crop_padded_100ep.json` |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8230 |
| Macro precision | 0.8414 |
| Macro recall | 0.8400 |
| Macro-F1 | 0.8384 |
| Weighted precision | 0.8258 |
| Weighted recall | 0.8230 |
| Weighted-F1 | 0.8227 |
| Reject precision | 0.7726 |
| Reject recall | 0.7750 |
| Reject-F1 | 0.7738 |
| Reject support | 320 |
| False accepts | 72 |
| False rejects | 73 |

## Per-Class Metrics

These per-class values are computed from `confusion_matrix_best_ensemble.csv` after applying the selected ensemble threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.7674 | 0.8250 | 0.7952 | 40 |
| Bengal | 0.8537 | 0.8750 | 0.8642 | 40 |
| Birman | 0.9250 | 0.9250 | 0.9250 | 40 |
| Bombay | 0.8837 | 0.9500 | 0.9157 | 40 |
| British Shorthair | 1.0000 | 0.8000 | 0.8889 | 40 |
| Maine Coon | 0.7949 | 0.7750 | 0.7848 | 40 |
| Ragdoll | 0.8500 | 0.8500 | 0.8500 | 40 |
| Sphynx | 0.9189 | 0.8500 | 0.8831 | 40 |
| Tabby | 0.6486 | 0.6000 | 0.6234 | 40 |
| Tiger Cat | 0.8788 | 0.7250 | 0.7945 | 40 |
| Beagle | 0.8140 | 0.8750 | 0.8434 | 40 |
| Pug | 0.9250 | 0.9250 | 0.9250 | 40 |
| Boxer | 0.9000 | 0.9000 | 0.9000 | 40 |
| Shiba Inu | 0.9118 | 0.7750 | 0.8378 | 40 |
| Samoyed | 0.8125 | 0.9750 | 0.8864 | 40 |
| Golden Retriever | 0.7297 | 0.9000 | 0.8060 | 30 |
| German Shepherd | 0.7333 | 0.7333 | 0.7333 | 30 |
| Siberian Husky | 0.8462 | 0.8684 | 0.8571 | 38 |
| Dalmatian | 0.8857 | 0.8378 | 0.8611 | 37 |
| Rottweiler | 0.8182 | 0.9000 | 0.8571 | 30 |
| reject | 0.7726 | 0.7750 | 0.7738 | 320 |

## Notes

- This table is threshold-calibrated.
- The ensemble uses weighted softmax probabilities from the three scratch models.
- The selected threshold and weights are chosen on the fixed internal validation split.
- Use the matching `confusion_matrix_best_ensemble.png` file as the report figure.
