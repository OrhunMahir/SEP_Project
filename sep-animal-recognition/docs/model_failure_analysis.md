# Model Failure Analysis

The failure-analysis workflow turns a labelled `predictions.csv` into a
report-ready diagnosis of model errors. It operates on saved predictions, so
it does not reload checkpoints, repeat inference, or change existing
evaluation results.

## Input

`scripts/evaluate_official_ensemble.py` already writes the required columns:

| Column | Description |
|---|---|
| `filename` | Image path relative to the evaluation image folder |
| `label` | Ground-truth external label (`-1` or `0` to `19`) |
| `prediction` | Predicted external label (`-1` or `0` to `19`) |
| `confidence` | Maximum ensemble probability before thresholding |

Rows and labels are validated before analysis. Empty filenames, duplicate
filenames, invalid labels, and confidence values outside `[0, 1]` are rejected
instead of silently entering the report.

## Usage

First run the labelled official evaluation. Then generate the failure report:

```bash
export PYTHONPATH="$PWD/src"

python scripts/generate_failure_report.py \
  --predictions runs/official_pretrained_50ep_ensemble_cpu/predictions.csv \
  --image-root /absolute/path/to/official_validation_images \
  --high-confidence-threshold 0.80 \
  --output-dir runs/official_pretrained_50ep_ensemble_cpu/failure_analysis
```

`--image-root` is optional. When it is provided, the report includes a contact
sheet of the highest-confidence failures whose source images are available.
The tabular analysis and plots do not require access to the images.

## Generated Artifacts

- `model_failure_report.md`: headline findings and linked figures.
- `failure_analysis_summary.json`: machine-readable summary and top findings.
- `failure_cases.csv`: every error ordered by confidence.
- `confusion_pairs.csv`: directed true-to-predicted error pairs.
- `per_class_failure_metrics.csv`: support, accuracy, and failure modes.
- `confidence_bins.csv`: reliability statistics and calibration gaps.
- `top_confusion_pairs.png`: most frequent directed class confusions.
- `per_class_accuracy.png`: represented classes ordered hardest first.
- `confidence_reliability.png`: observed accuracy versus mean confidence.
- `confidence_distribution.png`: correct and incorrect confidence histograms.
- `high_confidence_failures.png`: optional image contact sheet.

## Failure Categories

- **Class confusion:** one known animal class is predicted as another.
- **False accept:** a reject example is assigned to a known animal class.
- **False reject:** a known animal class is assigned to reject.

The categories are mutually exclusive, so their counts add up to the total
number of failures.

## Confidence Interpretation

High-confidence mistakes are important because the confidence threshold is
unlikely to remove them. They can expose systematic visual shortcuts or class
boundaries that need targeted data collection.

The reliability diagram groups predictions into equal-width confidence bins.
Points below the diagonal indicate overconfidence; points above it indicate
underconfidence. The JSON summary also reports expected calibration error
(ECE), weighted by the number of samples in each populated bin.

The confidence stored by the official evaluator is the maximum ensemble
probability before thresholding. A false reject may therefore have a low
maximum probability even though its final external prediction is `-1`.
