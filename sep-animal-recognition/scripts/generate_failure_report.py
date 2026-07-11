#!/usr/bin/env python3
"""Generate a detailed model failure report from a labelled predictions CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from animal_recognition.failure_analysis import (
    PredictionRecord,
    analysis_summary,
    confidence_bin_rows,
    confusion_pair_rows,
    failure_rows,
    parse_prediction_records,
    per_class_failure_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAILURE_FIELDS = (
    "filename",
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "confidence",
    "failure_type",
)
PAIR_FIELDS = (
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "count",
    "mean_confidence",
    "max_confidence",
)
PER_CLASS_FIELDS = (
    "label",
    "class_name",
    "support",
    "correct",
    "failures",
    "accuracy",
    "false_accepts",
    "false_rejects",
    "class_confusions",
    "mean_error_confidence",
)
CONFIDENCE_BIN_FIELDS = (
    "bin",
    "lower_bound",
    "upper_bound",
    "count",
    "accuracy",
    "mean_confidence",
    "calibration_gap",
)


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Load and validate the labelled prediction CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return parse_prediction_records(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Write a stable CSV schema even when there are no rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_pairs(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    limit: int,
) -> bool:
    """Plot the most frequent directed confusion pairs."""
    if not rows:
        return False
    import matplotlib.pyplot as plt

    selected = list(rows[:limit])
    labels = [
        f"{row['true_class']} -> {row['predicted_class']}" for row in selected
    ]
    counts = [int(row["count"]) for row in selected]
    figure_height = max(4.5, 0.42 * len(selected) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    positions = list(range(len(selected)))
    axis.barh(positions, counts, color="#b4473e")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Failure count")
    axis.set_title("Most frequent directed confusion pairs")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def plot_per_class_accuracy(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot represented classes from hardest to easiest."""
    import matplotlib.pyplot as plt

    labels = [str(row["class_name"]) for row in rows]
    values = [float(row["accuracy"]) for row in rows]
    figure_height = max(6.0, 0.32 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    positions = list(range(len(rows)))
    axis.barh(positions, values, color="#357a78")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Accuracy")
    axis.set_title("Per-class accuracy, hardest first")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_reliability_diagram(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot observed accuracy against mean confidence for populated bins."""
    import matplotlib.pyplot as plt

    populated = [row for row in rows if int(row["count"]) > 0]
    confidences = [float(row["mean_confidence"]) for row in populated]
    accuracies = [float(row["accuracy"]) for row in populated]
    sizes = [max(35, int(row["count"]) * 6) for row in populated]
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    axis.scatter(
        confidences,
        accuracies,
        s=sizes,
        color="#305f8d",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.7,
        label="Observed bins",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed accuracy")
    axis.set_title("Confidence reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confidence_distribution(
    path: Path,
    records: Sequence[PredictionRecord],
) -> None:
    """Compare confidence distributions for correct and incorrect predictions."""
    import matplotlib.pyplot as plt

    correct = [record.confidence for record in records if record.correct]
    failures = [record.confidence for record in records if not record.correct]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    bins = [index / 10 for index in range(11)]
    if correct:
        axis.hist(correct, bins=bins, alpha=0.65, label="Correct", color="#357a78")
    if failures:
        axis.hist(failures, bins=bins, alpha=0.65, label="Incorrect", color="#b4473e")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Prediction count")
    axis.set_title("Confidence distribution by outcome")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_failure_gallery(
    path: Path,
    image_root: Path,
    rows: Sequence[Mapping[str, object]],
    limit: int,
) -> tuple[int, list[str]]:
    """Create a report-ready contact sheet for the highest-confidence failures."""
    selected: list[tuple[Mapping[str, object], Image.Image]] = []
    missing: list[str] = []
    for row in rows[:limit]:
        image_path = image_root / str(row["filename"])
        if not image_path.is_file():
            missing.append(str(row["filename"]))
            continue
        try:
            with Image.open(image_path) as image:
                selected.append((row, image.convert("RGB").copy()))
        except OSError:
            missing.append(str(row["filename"]))

    if not selected:
        return 0, missing

    columns = min(4, len(selected))
    rows_count = math.ceil(len(selected) / columns)
    cell_width = 260
    image_height = 190
    caption_height = 62
    cell_height = image_height + caption_height
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows_count * cell_height),
        color=(245, 245, 242),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (row, image) in enumerate(selected):
        column = index % columns
        row_index = index // columns
        left = column * cell_width
        top = row_index * cell_height
        contained = ImageOps.contain(image, (cell_width - 16, image_height - 16))
        image_left = left + (cell_width - contained.width) // 2
        image_top = top + (image_height - contained.height) // 2
        canvas.paste(contained, (image_left, image_top))
        draw.rectangle(
            (left, top + image_height, left + cell_width, top + cell_height),
            fill=(32, 35, 38),
        )
        draw.text(
            (left + 8, top + image_height + 7),
            f"True: {row['true_class']}",
            fill=(255, 255, 255),
        )
        draw.text(
            (left + 8, top + image_height + 25),
            f"Pred: {row['predicted_class']}",
            fill=(255, 215, 120),
        )
        draw.text(
            (left + 8, top + image_height + 43),
            f"Confidence: {float(row['confidence']):.3f}",
            fill=(220, 220, 220),
        )
    canvas.save(path, quality=95)
    return len(selected), missing


def markdown_table_row(values: Sequence[object]) -> str:
    """Format one Markdown table row."""
    return "| " + " | ".join(str(value) for value in values) + " |"


def write_report(
    path: Path,
    predictions_path: Path,
    summary: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    per_class: Sequence[Mapping[str, object]],
    gallery_count: int,
    has_pair_plot: bool,
) -> None:
    """Write a report that links findings to generated evidence."""
    hardest = list(per_class[:5])
    high_threshold = float(summary["high_confidence_threshold"])
    high_confidence = [
        row for row in failures if float(row["confidence"]) >= high_threshold
    ][:10]
    lines = [
        "# Model Failure Analysis",
        "",
        "## Evaluation Summary",
        "",
        f"- Prediction source: `{predictions_path}`",
        f"- Labelled samples: {int(summary['samples'])}",
        f"- Accuracy: {float(summary['accuracy']):.4f}",
        f"- Total failures: {int(summary['failures'])}",
        (
            f"- High-confidence failures (confidence >= {high_threshold:.2f}): "
            f"{int(summary['high_confidence_failures'])}"
        ),
        f"- Expected calibration error: {float(summary['expected_calibration_error']):.4f}",
        "",
        "## Failure Breakdown",
        "",
        "| Failure type | Count |",
        "|---|---:|",
        markdown_table_row(("Class confusion", int(summary["class_confusions"]))),
        markdown_table_row(("False accept", int(summary["false_accepts"]))),
        markdown_table_row(("False reject", int(summary["false_rejects"]))),
        "",
        "## Hardest Classes",
        "",
        "| Class | Support | Failures | Accuracy | Mean error confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in hardest:
        lines.append(markdown_table_row((
            row["class_name"],
            row["support"],
            row["failures"],
            f"{float(row['accuracy']):.4f}",
            f"{float(row['mean_error_confidence']):.4f}",
        )))

    lines.extend([
        "",
        "## Most Frequent Confusion Pairs",
        "",
        "| True class | Predicted class | Count | Mean confidence |",
        "|---|---|---:|---:|",
    ])
    if pairs:
        for row in pairs[:10]:
            lines.append(markdown_table_row((
                row["true_class"],
                row["predicted_class"],
                row["count"],
                f"{float(row['mean_confidence']):.4f}",
            )))
    else:
        lines.append("| No incorrect predictions | - | 0 | - |")

    lines.extend([
        "",
        "## High-Confidence Failures",
        "",
        "| File | True class | Predicted class | Confidence | Failure type |",
        "|---|---|---|---:|---|",
    ])
    if high_confidence:
        for row in high_confidence:
            lines.append(markdown_table_row((
                f"`{row['filename']}`",
                row["true_class"],
                row["predicted_class"],
                f"{float(row['confidence']):.4f}",
                str(row["failure_type"]).replace("_", " "),
            )))
    else:
        lines.append("| None | - | - | - | - |")

    lines.extend([
        "",
        "## Diagnostic Figures",
        "",
        "![Per-class accuracy](per_class_accuracy.png)",
        "",
        "![Confidence reliability diagram](confidence_reliability.png)",
        "",
        "![Confidence distribution](confidence_distribution.png)",
    ])
    if has_pair_plot:
        lines.extend(["", "![Top confusion pairs](top_confusion_pairs.png)"])
    if gallery_count:
        lines.extend([
            "",
            "## Failure Gallery",
            "",
            (
                f"The gallery contains the {gallery_count} highest-confidence failures "
                "whose source images were available."
            ),
            "",
            "![High-confidence failure gallery](high_confidence_failures.png)",
        ])
    lines.extend([
        "",
        "## Reading the Report",
        "",
        (
            "High-confidence errors deserve priority because confidence thresholding "
            "is less likely to reject them. Repeated directed confusion pairs indicate "
            "specific class boundaries that may benefit from additional data or targeted "
            "augmentation. A reliability point below the diagonal indicates that the model "
            "is overconfident in that confidence range."
        ),
        "",
        "Full tables are available in `failure_cases.csv`, `confusion_pairs.csv`, "
        "`per_class_failure_metrics.csv`, and `confidence_bins.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root used to create a high-confidence failure gallery.",
    )
    parser.add_argument("--high-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--confidence-bins", type=int, default=10)
    parser.add_argument("--top-pairs", type=int, default=12)
    parser.add_argument("--gallery-limit", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.predictions.is_file():
        raise FileNotFoundError(f"Predictions CSV was not found: {args.predictions}")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        raise ValueError("--high-confidence-threshold must be between 0 and 1.")
    if args.confidence_bins <= 0 or args.top_pairs <= 0 or args.gallery_limit <= 0:
        raise ValueError("Bin, pair, and gallery limits must be positive.")
    if args.image_root is not None and not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root was not found: {args.image_root}")

    output_dir = args.output_dir or args.predictions.parent / "failure_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_predictions(args.predictions)
    failures = failure_rows(records)
    pairs = confusion_pair_rows(records)
    per_class = per_class_failure_rows(records)
    confidence_bins = confidence_bin_rows(records, args.confidence_bins)
    summary = analysis_summary(
        records,
        confidence_bins,
        args.high_confidence_threshold,
    )

    write_csv(output_dir / "failure_cases.csv", failures, FAILURE_FIELDS)
    write_csv(output_dir / "confusion_pairs.csv", pairs, PAIR_FIELDS)
    write_csv(
        output_dir / "per_class_failure_metrics.csv",
        per_class,
        PER_CLASS_FIELDS,
    )
    write_csv(
        output_dir / "confidence_bins.csv",
        confidence_bins,
        CONFIDENCE_BIN_FIELDS,
    )
    has_pair_plot = plot_confusion_pairs(
        output_dir / "top_confusion_pairs.png",
        pairs,
        args.top_pairs,
    )
    plot_per_class_accuracy(output_dir / "per_class_accuracy.png", per_class)
    plot_reliability_diagram(output_dir / "confidence_reliability.png", confidence_bins)
    plot_confidence_distribution(output_dir / "confidence_distribution.png", records)

    gallery_count = 0
    missing_gallery_images: list[str] = []
    if args.image_root is not None:
        high_confidence_failures = [
            row
            for row in failures
            if float(row["confidence"]) >= args.high_confidence_threshold
        ]
        gallery_count, missing_gallery_images = create_failure_gallery(
            output_dir / "high_confidence_failures.png",
            args.image_root,
            high_confidence_failures,
            args.gallery_limit,
        )

    summary_payload = {
        **summary,
        "predictions": str(args.predictions),
        "image_root": str(args.image_root) if args.image_root is not None else None,
        "represented_classes": len(per_class),
        "confusion_pairs": len(pairs),
        "gallery_images": gallery_count,
        "missing_gallery_images": missing_gallery_images,
        "hardest_classes": per_class[:5],
        "top_confusion_pairs": pairs[:args.top_pairs],
    }
    (output_dir / "failure_analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir / "model_failure_report.md",
        args.predictions,
        summary,
        failures,
        pairs,
        per_class,
        gallery_count,
        has_pair_plot,
    )
    print(f"Saved failure analysis report to: {output_dir}")


if __name__ == "__main__":
    main()
