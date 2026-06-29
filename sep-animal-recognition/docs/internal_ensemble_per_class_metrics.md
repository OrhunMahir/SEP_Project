# Internal Validation Per-Class Metrics for Best Ensembles

These tables report per-class metrics for the two best ensemble models on the internal validation split used for model selection and threshold calibration. This is not the instructor-provided official test/evaluation set.

Per-class `Accuracy` is computed as one-vs-rest accuracy: `(TP + TN) / all validation samples`. `Recall` is the class-wise hit rate: `TP / support`.

## Run Settings

| Ensemble | Validation samples | Models | Weights | Threshold | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Pretrained 50ep ensemble | 1085 | ResNet-18 pretrained + EfficientNet-B0 pretrained + Swin-Tiny pretrained | `0.35 / 0.35 / 0.30` | `0.30` | 0.9548 | 0.9595 | 0.9547 |
| Scratch 100ep ensemble | 1085 | Custom CNN 8-conv scratch + ResNet-18 scratch + EfficientNet-B0 scratch | `0.45 / 0.25 / 0.30` | `0.32` | 0.8230 | 0.8384 | 0.8227 |

## Pretrained 50ep ensemble Per-Class Metrics

| Class | Accuracy | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abyssinian | 0.9972 | 0.9302 | 1.0000 | 0.9639 | 40 |
| Bengal | 0.9972 | 0.9512 | 0.9750 | 0.9630 | 40 |
| Birman | 0.9982 | 0.9524 | 1.0000 | 0.9756 | 40 |
| Bombay | 0.9972 | 0.9512 | 0.9750 | 0.9630 | 40 |
| British Shorthair | 0.9963 | 0.9737 | 0.9250 | 0.9487 | 40 |
| Maine Coon | 0.9982 | 0.9750 | 0.9750 | 0.9750 | 40 |
| Ragdoll | 0.9991 | 1.0000 | 0.9750 | 0.9873 | 40 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40 |
| Tabby | 0.9945 | 0.9474 | 0.9000 | 0.9231 | 40 |
| Tiger Cat | 0.9963 | 0.9500 | 0.9500 | 0.9500 | 40 |
| Beagle | 0.9954 | 0.9070 | 0.9750 | 0.9398 | 40 |
| Pug | 0.9972 | 0.9512 | 0.9750 | 0.9630 | 40 |
| Boxer | 0.9982 | 0.9524 | 1.0000 | 0.9756 | 40 |
| Shiba Inu | 0.9972 | 0.9512 | 0.9750 | 0.9630 | 40 |
| Samoyed | 0.9954 | 0.8889 | 1.0000 | 0.9412 | 40 |
| Golden Retriever | 0.9982 | 0.9667 | 0.9667 | 0.9667 | 30 |
| German Shepherd | 0.9963 | 0.9333 | 0.9333 | 0.9333 | 30 |
| Siberian Husky | 0.9963 | 0.9250 | 0.9737 | 0.9487 | 38 |
| Dalmatian | 0.9972 | 0.9474 | 0.9730 | 0.9600 | 37 |
| Rottweiler | 0.9982 | 0.9375 | 1.0000 | 0.9677 | 30 |
| reject | 0.9659 | 0.9701 | 0.9125 | 0.9404 | 320 |

## Scratch 100ep ensemble Per-Class Metrics

| Class | Accuracy | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abyssinian | 0.9843 | 0.7674 | 0.8250 | 0.7952 | 40 |
| Bengal | 0.9899 | 0.8537 | 0.8750 | 0.8642 | 40 |
| Birman | 0.9945 | 0.9250 | 0.9250 | 0.9250 | 40 |
| Bombay | 0.9935 | 0.8837 | 0.9500 | 0.9157 | 40 |
| British Shorthair | 0.9926 | 1.0000 | 0.8000 | 0.8889 | 40 |
| Maine Coon | 0.9843 | 0.7949 | 0.7750 | 0.7848 | 40 |
| Ragdoll | 0.9889 | 0.8500 | 0.8500 | 0.8500 | 40 |
| Sphynx | 0.9917 | 0.9189 | 0.8500 | 0.8831 | 40 |
| Tabby | 0.9733 | 0.6486 | 0.6000 | 0.6234 | 40 |
| Tiger Cat | 0.9862 | 0.8788 | 0.7250 | 0.7945 | 40 |
| Beagle | 0.9880 | 0.8140 | 0.8750 | 0.8434 | 40 |
| Pug | 0.9945 | 0.9250 | 0.9250 | 0.9250 | 40 |
| Boxer | 0.9926 | 0.9000 | 0.9000 | 0.9000 | 40 |
| Shiba Inu | 0.9889 | 0.9118 | 0.7750 | 0.8378 | 40 |
| Samoyed | 0.9908 | 0.8125 | 0.9750 | 0.8864 | 40 |
| Golden Retriever | 0.9880 | 0.7297 | 0.9000 | 0.8060 | 30 |
| German Shepherd | 0.9853 | 0.7333 | 0.7333 | 0.7333 | 30 |
| Siberian Husky | 0.9899 | 0.8462 | 0.8684 | 0.8571 | 38 |
| Dalmatian | 0.9908 | 0.8857 | 0.8378 | 0.8611 | 37 |
| Rottweiler | 0.9917 | 0.8182 | 0.9000 | 0.8571 | 30 |
| reject | 0.8664 | 0.7726 | 0.7750 | 0.7738 | 320 |

## Source Files

The per-class metrics above were computed from the selected best-ensemble confusion matrices:

```text
/Users/orhun/Desktop/SEP_Project/Confmatrix/Ensemble/Pretrained_50ep_Results/ensemble_resnet18_efficientnet_b0_swin_tiny_pretrained_padded_50ep/confusion_matrix_best_ensemble.csv
/Users/orhun/Desktop/SEP_Project/Confmatrix/Ensemble/Scratch_100ep_Results/ensemble_custom_cnn_resnet18_efficientnet_b0_scratch_padded_100ep/confusion_matrix_best_ensemble.csv
```

The aggregate best-ensemble metrics are documented in `docs/internal_ensemble_validation_results.md`.
