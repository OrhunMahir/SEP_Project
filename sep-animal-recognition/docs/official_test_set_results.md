# Official Test Set Results

These results were computed on the instructor-provided labelled evaluation image set:

```text
~/projects/SEP_Project_code/official_validation_images
```

This set contains `143` images and a `labels.csv` file. It is separate from the internal train/validation split used for model selection and threshold tuning. The fixed ensemble weights and thresholds selected on internal validation were applied here without retuning.

The project requirement states that the average inference time must not exceed 5 seconds per image. Both runs below satisfy this requirement on CPU.

## Summary

| Setup | Accuracy | Macro-F1 | Weighted-F1 | Reject-F1 | False accepts | False rejects | Avg sec/image | Max sec/image | Under 5 sec avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained 50ep ensemble | 0.7902 | 0.7695 | 0.7948 | 0.8605 | 3 | 9 | 0.1587 | 3.5620 | True |
| Scratch 100ep ensemble | 0.6853 | 0.6396 | 0.6693 | 0.7451 | 2 | 24 | 0.1339 | 3.7635 | True |

## Pretrained 50ep Ensemble

| Setting | Value |
|---|---:|
| Models | ResNet-18 pretrained + EfficientNet-B0 pretrained + Swin-Tiny pretrained |
| Checkpoint epochs | 47 / 30 / 29 |
| Weights | `0.35 / 0.35 / 0.30` |
| Threshold | `0.30` |
| Preprocessing | YOLO padded crop + raw fallback |
| Device for this run | `cpu` |
| Images | 143 |

| Metric | Value |
|---|---:|
| Accuracy | 0.7902 |
| Macro precision | 0.8203 |
| Macro recall | 0.7448 |
| Macro-F1 | 0.7695 |
| Weighted precision | 0.8210 |
| Weighted recall | 0.7902 |
| Weighted-F1 | 0.7948 |
| Reject precision | 0.8043 |
| Reject recall | 0.9250 |
| Reject-F1 | 0.8605 |
| False accepts | 3 |
| False rejects | 9 |
| Avg seconds/image | 0.1587 |
| Median seconds/image | 0.1316 |
| Max seconds/image | 3.5620 |
| Under 5 sec average | True |

### Pretrained Per-Class Precision / Recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.8000 | 1.0000 | 0.8889 | 4 |
| Bengal | 0.8333 | 1.0000 | 0.9091 | 5 |
| Birman | 0.5000 | 0.5000 | 0.5000 | 4 |
| Bombay | 0.8333 | 1.0000 | 0.9091 | 5 |
| British Shorthair | 1.0000 | 0.6000 | 0.7500 | 5 |
| Maine Coon | 0.8000 | 0.8000 | 0.8000 | 5 |
| Ragdoll | 0.4286 | 0.5000 | 0.4615 | 6 |
| Sphynx | 1.0000 | 1.0000 | 1.0000 | 5 |
| Tabby | 0.3333 | 0.4000 | 0.3636 | 5 |
| Tiger Cat | 0.1429 | 0.2000 | 0.1667 | 5 |
| Beagle | 1.0000 | 0.8333 | 0.9091 | 6 |
| Pug | 1.0000 | 0.7500 | 0.8571 | 4 |
| Boxer | 1.0000 | 1.0000 | 1.0000 | 5 |
| Shiba Inu | 1.0000 | 0.6000 | 0.7500 | 5 |
| Samoyed | 1.0000 | 0.8333 | 0.9091 | 6 |
| Golden Retriever | 1.0000 | 0.6667 | 0.8000 | 6 |
| German Shepherd | 0.7500 | 0.6000 | 0.6667 | 5 |
| Siberian Husky | 1.0000 | 0.6000 | 0.7500 | 5 |
| Dalmatian | 1.0000 | 1.0000 | 1.0000 | 6 |
| Rottweiler | 1.0000 | 0.8333 | 0.9091 | 6 |
| reject | 0.8043 | 0.9250 | 0.8605 | 40 |

Files produced by this run:

```text
runs/official_pretrained_50ep_ensemble_cpu/timing_summary.json
runs/official_pretrained_50ep_ensemble_cpu/predictions.csv
runs/official_pretrained_50ep_ensemble_cpu/per_class_metrics.csv
runs/official_pretrained_50ep_ensemble_cpu/confusion_matrix.csv
runs/official_pretrained_50ep_ensemble_cpu/confusion_matrix.png
```

## Scratch 100ep Ensemble

| Setting | Value |
|---|---:|
| Models | Custom CNN 8-conv scratch + ResNet-18 scratch + EfficientNet-B0 scratch |
| Checkpoint epochs | 84 / 85 / 94 |
| Weights | `0.45 / 0.25 / 0.30` |
| Threshold | `0.32` |
| Preprocessing | YOLO padded crop + raw fallback |
| Device for this run | `cpu` |
| Images | 143 |

| Metric | Value |
|---|---:|
| Accuracy | 0.6853 |
| Macro precision | 0.7797 |
| Macro recall | 0.5968 |
| Macro-F1 | 0.6396 |
| Weighted precision | 0.7434 |
| Weighted recall | 0.6853 |
| Weighted-F1 | 0.6693 |
| Reject precision | 0.6129 |
| Reject recall | 0.9500 |
| Reject-F1 | 0.7451 |
| False accepts | 2 |
| False rejects | 24 |
| Avg seconds/image | 0.1339 |
| Median seconds/image | 0.1058 |
| Max seconds/image | 3.7635 |
| Under 5 sec average | True |

### Scratch Per-Class Precision / Recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abyssinian | 0.8000 | 1.0000 | 0.8889 | 4 |
| Bengal | 0.8333 | 1.0000 | 0.9091 | 5 |
| Birman | 0.6667 | 0.5000 | 0.5714 | 4 |
| Bombay | 0.8333 | 1.0000 | 0.9091 | 5 |
| British Shorthair | 1.0000 | 0.4000 | 0.5714 | 5 |
| Maine Coon | 0.7500 | 0.6000 | 0.6667 | 5 |
| Ragdoll | 0.4444 | 0.6667 | 0.5333 | 6 |
| Sphynx | 0.8000 | 0.8000 | 0.8000 | 5 |
| Tabby | 0.5000 | 0.4000 | 0.4444 | 5 |
| Tiger Cat | 0.2000 | 0.2000 | 0.2000 | 5 |
| Beagle | 1.0000 | 0.5000 | 0.6667 | 6 |
| Pug | 1.0000 | 0.2500 | 0.4000 | 4 |
| Boxer | 1.0000 | 0.4000 | 0.5714 | 5 |
| Shiba Inu | 0.6000 | 0.6000 | 0.6000 | 5 |
| Samoyed | 1.0000 | 0.6667 | 0.8000 | 6 |
| Golden Retriever | 1.0000 | 0.5000 | 0.6667 | 6 |
| German Shepherd | 1.0000 | 0.4000 | 0.5714 | 5 |
| Siberian Husky | 0.3333 | 0.2000 | 0.2500 | 5 |
| Dalmatian | 1.0000 | 1.0000 | 1.0000 | 6 |
| Rottweiler | 1.0000 | 0.5000 | 0.6667 | 6 |
| reject | 0.6129 | 0.9500 | 0.7451 | 40 |

Files produced by this run:

```text
runs/official_scratch_100ep_ensemble_cpu/timing_summary.json
runs/official_scratch_100ep_ensemble_cpu/predictions.csv
runs/official_scratch_100ep_ensemble_cpu/per_class_metrics.csv
runs/official_scratch_100ep_ensemble_cpu/confusion_matrix.csv
runs/official_scratch_100ep_ensemble_cpu/confusion_matrix.png
```
