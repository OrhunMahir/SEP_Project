# Internal Ensemble Validation Results

These results come from the internal validation split used for model selection
and threshold calibration. The split contains `1085` labelled images and is
separate from the instructor-provided official labelled evaluation set.

Both ensembles use the same preprocessing pipeline:

- YOLO padded square crop with `10%` padding
- raw image fallback when YOLO does not detect a target cat/dog
- confidence-threshold based reject decision

## Summary

| Ensemble | Validation samples | Models | Weights | Threshold | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted precision | Weighted recall | Weighted-F1 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained 50ep ensemble | 1085 | ResNet-18 + EfficientNet-B0 + Swin-Tiny | `0.35 / 0.35 / 0.30` | `0.30` | 0.9548 | 0.9506 | 0.9695 | 0.9595 | 0.9558 | 0.9548 | 0.9547 |
| Scratch 100ep ensemble | 1085 | Custom CNN 8-conv + ResNet-18 + EfficientNet-B0 | `0.45 / 0.25 / 0.30` | `0.32` | 0.8230 | 0.8414 | 0.8400 | 0.8384 | 0.8258 | 0.8230 | 0.8227 |

## Reject Metrics

| Ensemble | Reject precision | Reject recall | Reject-F1 | Reject support | False accepts | False rejects |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained 50ep ensemble | 0.9701 | 0.9125 | 0.9404 | 320 | 28 | 9 |
| Scratch 100ep ensemble | 0.7726 | 0.7750 | 0.7738 | 320 | 72 | 73 |

## Reproduction

Run `scripts/evaluate_ensemble.py` as documented in the submission README.
Each new run writes its complete candidate grid, selected `best_ensemble`,
confusion matrix, and per-class outputs to the requested project-relative
`runs/` directory.
