# Pretrained 50 Epoch Ensemble Results

This document summarizes the threshold-calibrated pretrained ensemble that combines the 50-epoch ResNet-18, EfficientNet-B0, and Swin Tiny runs.

## Ensemble Summary

| Metric | Value |
|---|---:|
| Selection metric | `accuracy` |
| Best threshold | 0.3000 |
| Validation samples | 1085 |
| Device | `cuda` |

## Ensemble Members

| Index | Model | Weight | Checkpoint epoch | Config |
|---:|---|---:|---:|---|
| 0 | ResNet-18 pretrained | 0.3500 | 47 | `configs/resnet18_pretrained_yolo_crop_padded_50ep.json` |
| 1 | EfficientNet-B0 pretrained | 0.3500 | 30 | `configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json` |
| 2 | Swin Tiny pretrained | 0.3000 | 29 | `configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json` |

## Threshold-Calibrated Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9548 |
| Macro precision | 0.9506 |
| Macro recall | 0.9695 |
| Macro-F1 | 0.9595 |
| Weighted precision | 0.9558 |
| Weighted recall | 0.9548 |
| Weighted-F1 | 0.9547 |
| Reject precision | 0.9701 |
| Reject recall | 0.9125 |
| Reject-F1 | 0.9404 |
| Reject support | 320 |
| False accepts | 28 |
| False rejects | 9 |

## Per-Class Metrics

These per-class values are computed from `confusion_matrix_best_ensemble.csv` after applying the selected ensemble threshold.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.9302 | 1.0000 | 0.9639 | 40 |
| Bengal | 0.9512 | 0.9750 | 0.9630 | 40 |
| Birman | 0.9524 | 1.0000 | 0.9756 | 40 |
| Bombay | 0.9512 | 0.9750 | 0.9630 | 40 |
| British Shorthair | 0.9737 | 0.9250 | 0.9487 | 40 |
| Maine Coon | 0.9750 | 0.9750 | 0.9750 | 40 |
| Ragdoll | 1.0000 | 0.9750 | 0.9873 | 40 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 40 |
| Tabby | 0.9474 | 0.9000 | 0.9231 | 40 |
| Tiger Cat | 0.9500 | 0.9500 | 0.9500 | 40 |
| Beagle | 0.9070 | 0.9750 | 0.9398 | 40 |
| Pug | 0.9512 | 0.9750 | 0.9630 | 40 |
| Boxer | 0.9524 | 1.0000 | 0.9756 | 40 |
| Shiba Inu | 0.9512 | 0.9750 | 0.9630 | 40 |
| Samoyed | 0.8889 | 1.0000 | 0.9412 | 40 |
| Golden Retriever | 0.9667 | 0.9667 | 0.9667 | 30 |
| German Shepherd | 0.9333 | 0.9333 | 0.9333 | 30 |
| Siberian Husky | 0.9250 | 0.9737 | 0.9487 | 38 |
| Dalmatian | 0.9474 | 0.9730 | 0.9600 | 37 |
| Rottweiler | 0.9375 | 1.0000 | 0.9677 | 30 |
| reject | 0.9701 | 0.9125 | 0.9404 | 320 |

## Notes

- This table is threshold-calibrated.
- The ensemble uses weighted softmax probabilities from the three pretrained models.
- The selected threshold and weights are chosen on the fixed internal validation split.
- The confusion matrix source is `confusion_matrix_best_ensemble.csv`.
