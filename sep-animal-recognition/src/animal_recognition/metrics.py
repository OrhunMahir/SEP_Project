"""Metrics shared by baseline and later candidate-model experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .constants import CLASSES, NUM_OUTPUTS, REJECT_EXTERNAL, REJECT_INTERNAL

CONFUSION_LABELS = tuple(CLASSES) + ("reject",)


def classification_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    """Compute overall, macro, weighted, and reject-specific classification metrics."""
    labels = list(range(NUM_OUTPUTS))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="weighted", zero_division=0
    )
    reject_precision, reject_recall, reject_f1, reject_support = precision_recall_fscore_support(
        targets, predictions, labels=[REJECT_INTERNAL], average=None, zero_division=0
    )

    # Reject hatalarını ayrı sayarak modelin güvenlik davranışını görünür kılıyorum.
    false_accepts = sum(true == REJECT_INTERNAL and predicted != REJECT_INTERNAL
                        for true, predicted in zip(targets, predictions, strict=True))
    false_rejects = sum(true != REJECT_INTERNAL and predicted == REJECT_INTERNAL
                        for true, predicted in zip(targets, predictions, strict=True))

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "reject_precision": float(reject_precision[0]),
        "reject_recall": float(reject_recall[0]),
        "reject_f1": float(reject_f1[0]),
        "reject_support": int(reject_support[0]),
        "false_accepts": int(false_accepts),
        "false_rejects": int(false_rejects),
    }


def confusion_matrix_counts(targets: Sequence[int], predictions: Sequence[int]) -> list[list[int]]:
    """Return a stable 21x21 confusion matrix with rows=true and columns=predicted."""
    matrix = confusion_matrix(targets, predictions, labels=list(range(NUM_OUTPUTS)))
    return matrix.astype(int).tolist()


def write_confusion_matrix_csv(
    path: Path,
    targets: Sequence[int],
    predictions: Sequence[int],
) -> None:
    """Write confusion matrix counts as a labelled CSV table."""
    matrix = confusion_matrix_counts(targets, predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *CONFUSION_LABELS])
        for label, row in zip(CONFUSION_LABELS, matrix, strict=True):
            writer.writerow([label, *row])


def write_confusion_matrix_png(
    path: Path,
    targets: Sequence[int],
    predictions: Sequence[int],
    title: str,
) -> None:
    """Render a report-ready confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    matrix = confusion_matrix_counts(targets, predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(NUM_OUTPUTS))
    axis.set_yticks(range(NUM_OUTPUTS))
    axis.set_xticklabels(CONFUSION_LABELS, rotation=90, fontsize=7)
    axis.set_yticklabels(CONFUSION_LABELS, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def per_class_metrics(targets: Sequence[int], predictions: Sequence[int]) -> list[dict[str, float | int | str]]:
    """Compute precision, recall, F1, and support for each output class."""
    labels = list(range(NUM_OUTPUTS))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )

    records: list[dict[str, float | int | str]] = []
    for label in labels:
        external_label = REJECT_EXTERNAL if label == REJECT_INTERNAL else label
        class_name = "reject" if label == REJECT_INTERNAL else CLASSES[label]
        records.append({
            "internal_label": label,
            "external_label": external_label,
            "class_name": class_name,
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        })
    return records
