"""Reusable failure-analysis summaries for labelled prediction rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .constants import CLASSES, REJECT_EXTERNAL

VALID_EXTERNAL_LABELS = frozenset({REJECT_EXTERNAL, *range(len(CLASSES))})


@dataclass(frozen=True)
class PredictionRecord:
    """One labelled prediction with confidence."""

    filename: str
    label: int
    prediction: int
    confidence: float

    @property
    def correct(self) -> bool:
        return self.label == self.prediction

    @property
    def failure_type(self) -> str:
        if self.correct:
            return "correct"
        if self.label == REJECT_EXTERNAL:
            return "false_accept"
        if self.prediction == REJECT_EXTERNAL:
            return "false_reject"
        return "class_confusion"


def label_name(label: int) -> str:
    """Return a display name for one external label."""
    if label == REJECT_EXTERNAL:
        return "reject"
    if label not in VALID_EXTERNAL_LABELS:
        raise ValueError(f"Unsupported external label: {label}")
    return CLASSES[label]


def parse_prediction_records(rows: Iterable[Mapping[str, str]]) -> list[PredictionRecord]:
    """Validate CSV-like rows and convert values to typed records."""
    records: list[PredictionRecord] = []
    seen_filenames: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        missing = {"filename", "label", "prediction", "confidence"} - set(row)
        if missing:
            raise ValueError(
                f"Prediction row {row_number} is missing columns: {', '.join(sorted(missing))}."
            )
        filename = row["filename"].strip()
        if not filename:
            raise ValueError(f"Prediction row {row_number} has an empty filename.")
        if filename in seen_filenames:
            raise ValueError(f"Duplicate prediction filename: {filename}")
        seen_filenames.add(filename)

        label = int(row["label"])
        prediction = int(row["prediction"])
        confidence = float(row["confidence"])
        if label not in VALID_EXTERNAL_LABELS:
            raise ValueError(f"Invalid label {label} on row {row_number}.")
        if prediction not in VALID_EXTERNAL_LABELS:
            raise ValueError(f"Invalid prediction {prediction} on row {row_number}.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1 on row {row_number}.")
        records.append(PredictionRecord(filename, label, prediction, confidence))
    if not records:
        raise ValueError("Prediction file contains no rows.")
    return records


def failure_rows(records: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    """Return all failures ordered by confidence, highest first."""
    failures = [record for record in records if not record.correct]
    failures.sort(key=lambda record: (-record.confidence, record.filename))
    return [
        {
            "filename": record.filename,
            "true_label": record.label,
            "true_class": label_name(record.label),
            "predicted_label": record.prediction,
            "predicted_class": label_name(record.prediction),
            "confidence": record.confidence,
            "failure_type": record.failure_type,
        }
        for record in failures
    ]


def confusion_pair_rows(records: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    """Aggregate directed true-to-predicted pairs for incorrect predictions."""
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        if not record.correct:
            grouped[(record.label, record.prediction)].append(record.confidence)
    rows = [
        {
            "true_label": true_label,
            "true_class": label_name(true_label),
            "predicted_label": predicted_label,
            "predicted_class": label_name(predicted_label),
            "count": len(confidences),
            "mean_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
        }
        for (true_label, predicted_label), confidences in grouped.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            -float(row["mean_confidence"]),
            str(row["true_class"]),
            str(row["predicted_class"]),
        )
    )
    return rows


def per_class_failure_rows(records: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    """Summarize accuracy and failure modes for every represented true class."""
    grouped: dict[int, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    rows: list[dict[str, object]] = []
    for label in sorted(grouped):
        class_records = grouped[label]
        correct = sum(record.correct for record in class_records)
        failures = len(class_records) - correct
        error_confidences = [record.confidence for record in class_records if not record.correct]
        rows.append({
            "label": label,
            "class_name": label_name(label),
            "support": len(class_records),
            "correct": correct,
            "failures": failures,
            "accuracy": correct / len(class_records),
            "false_accepts": sum(
                record.failure_type == "false_accept" for record in class_records
            ),
            "false_rejects": sum(
                record.failure_type == "false_reject" for record in class_records
            ),
            "class_confusions": sum(
                record.failure_type == "class_confusion" for record in class_records
            ),
            "mean_error_confidence": (
                sum(error_confidences) / len(error_confidences)
                if error_confidences
                else 0.0
            ),
        })
    rows.sort(key=lambda row: (float(row["accuracy"]), -int(row["support"]), str(row["class_name"])))
    return rows


def confidence_bin_rows(
    records: Sequence[PredictionRecord],
    num_bins: int = 10,
) -> list[dict[str, object]]:
    """Build equal-width confidence bins for calibration inspection."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    bins: list[list[PredictionRecord]] = [[] for _ in range(num_bins)]
    for record in records:
        index = min(int(record.confidence * num_bins), num_bins - 1)
        bins[index].append(record)

    rows: list[dict[str, object]] = []
    for index, bin_records in enumerate(bins):
        count = len(bin_records)
        accuracy = (
            sum(record.correct for record in bin_records) / count if count else 0.0
        )
        mean_confidence = (
            sum(record.confidence for record in bin_records) / count if count else 0.0
        )
        rows.append({
            "bin": index + 1,
            "lower_bound": index / num_bins,
            "upper_bound": (index + 1) / num_bins,
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "calibration_gap": abs(mean_confidence - accuracy) if count else 0.0,
        })
    return rows


def analysis_summary(
    records: Sequence[PredictionRecord],
    confidence_bins: Sequence[Mapping[str, object]],
    high_confidence_threshold: float,
) -> dict[str, object]:
    """Compute headline failure counts and expected calibration error."""
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be between 0 and 1.")
    failures = [record for record in records if not record.correct]
    total = len(records)
    expected_calibration_error = sum(
        int(row["count"]) / total * float(row["calibration_gap"])
        for row in confidence_bins
    )
    return {
        "samples": total,
        "correct": total - len(failures),
        "failures": len(failures),
        "accuracy": (total - len(failures)) / total,
        "false_accepts": sum(
            record.failure_type == "false_accept" for record in failures
        ),
        "false_rejects": sum(
            record.failure_type == "false_reject" for record in failures
        ),
        "class_confusions": sum(
            record.failure_type == "class_confusion" for record in failures
        ),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_failures": sum(
            record.confidence >= high_confidence_threshold for record in failures
        ),
        "mean_failure_confidence": (
            sum(record.confidence for record in failures) / len(failures)
            if failures
            else 0.0
        ),
        "expected_calibration_error": expected_calibration_error,
    }
