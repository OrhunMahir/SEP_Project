"""Metrics shared by baseline and later candidate-model experiments."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .constants import CLASSES, NUM_OUTPUTS, REJECT_EXTERNAL, REJECT_INTERNAL


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