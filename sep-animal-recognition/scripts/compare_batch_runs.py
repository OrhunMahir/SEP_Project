#!/usr/bin/env python3
"""Collect batch-size sweep summary metrics into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = (
    "runs/custom_cnn_batch_16",
    "runs/custom_cnn_batch_32",
    "runs/custom_cnn_batch_64",
)


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_summary(run_dir: Path) -> dict:
    """Read one run summary file."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def row_from_summary(run_dir: Path, summary: dict) -> dict[str, float | int | str]:
    """Extract report-friendly fields from one training summary."""
    best_metrics = summary.get("best_validation_metrics", {})
    return {
        "run_dir": str(run_dir),
        "experiment_name": str(summary.get("experiment_name", "")),
        "epochs_completed": int(summary.get("epochs_completed", 0)),
        "best_validation_macro_f1": float(summary.get("best_validation_macro_f1", 0.0)),
        "accuracy": float(best_metrics.get("accuracy", 0.0)),
        "macro_precision": float(best_metrics.get("macro_precision", 0.0)),
        "macro_recall": float(best_metrics.get("macro_recall", 0.0)),
        "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(best_metrics.get("weighted_f1", 0.0)),
        "reject_precision": float(best_metrics.get("reject_precision", 0.0)),
        "reject_recall": float(best_metrics.get("reject_recall", 0.0)),
        "reject_f1": float(best_metrics.get("reject_f1", 0.0)),
        "false_accepts": int(best_metrics.get("false_accepts", 0)),
        "false_rejects": int(best_metrics.get("false_rejects", 0)),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write comparison rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="Run directories containing summary.json files.",
    )
    parser.add_argument(
        "--output",
        default="runs/batch_size_sweep_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = []
    for run_text in args.runs:
        run_dir = resolve_project_path(run_text)
        rows.append(row_from_summary(run_dir, read_summary(run_dir)))

    output_path = resolve_project_path(args.output)
    write_csv(output_path, rows)
    print(f"Saved batch-size comparison to: {output_path}")


if __name__ == "__main__":
    main()