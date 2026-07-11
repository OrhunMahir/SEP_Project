# Robustness Benchmark

This benchmark measures how quickly validation performance degrades when input
quality moves away from the clean data distribution. It is an evaluation-only
workflow: source images, checkpoints, training configurations, and previously
reported results are not modified.

## Corruptions

Six deterministic corruption families are evaluated at five severity levels:

| Corruption | Simulated condition |
|---|---|
| Gaussian noise | Sensor noise and low-light grain |
| Gaussian blur | Defocus and camera motion |
| Brightness reduction | Poor illumination |
| Contrast reduction | Flat or hazy capture |
| JPEG compression | Low-bandwidth image transfer |
| Central occlusion | Partial obstruction of the animal |

Severity 1 is mild and severity 5 is strong. The clean validation result is
always recorded first as severity 0. Random noise uses a stable per-image seed,
so repeated runs with the same arguments are directly comparable.

## Final Ensemble Example

Run from the `sep-animal-recognition` directory:

```bash
export PYTHONPATH="$PWD/src"

python scripts/evaluate_robustness.py \
  --config configs/resnet18_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/resnet18_pretrained_yolo_crop_padded_50ep/best.pt \
  --config configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/efficientnet_b0_pretrained_yolo_crop_padded_50ep/best.pt \
  --config configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json \
  --checkpoint runs/swin_tiny_pretrained_yolo_crop_padded_50ep/best.pt \
  --weights 0.35,0.35,0.30 \
  --threshold 0.30 \
  --device cuda \
  --output-dir runs/pretrained_50ep_robustness
```

For a quick pipeline check, add `--max-samples 32 --severities 1`. A full run
should use the complete fixed validation split and all five severity levels.

## Generated Artifacts

The output directory contains:

- `robustness_results.csv`: metrics for every corruption and severity.
- `robustness_summary.json`: run parameters, model inputs, and full results.
- `accuracy_by_severity.png`: accuracy degradation curves.
- `macro_f1_by_severity.png`: Macro-F1 degradation curves.
- `model_robustness_report.md`: a generated, report-ready summary.

Accuracy, Macro-F1, reject F1, false accepts, and false rejects use the same
shared metric implementation as the project's existing evaluation scripts.
Percentage-point drops are computed against the clean baseline from the same
run.

## Interpretation

The benchmark is diagnostic rather than a replacement for clean validation.
The most useful comparisons are:

1. Clean versus severity-1 performance, which shows sensitivity to small input
   changes.
2. The slope across severities, which shows how gracefully performance
   degrades.
3. False-accept and false-reject changes, which show whether confidence
   thresholding remains reliable as image quality falls.

Corruptions are synthetic and do not represent every real camera or lighting
condition. Conclusions should therefore be framed as controlled stress-test
results, not as a guarantee of field performance.
