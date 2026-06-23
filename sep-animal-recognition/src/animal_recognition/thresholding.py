"""Confidence-threshold decisions shared by evaluation and final inference."""

from __future__ import annotations

import torch

from .constants import REJECT_INTERNAL


def apply_confidence_threshold(
    probabilities: torch.Tensor,
    threshold: float,
    reject_label: int = REJECT_INTERNAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return class predictions after replacing low-confidence cases with reject."""
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must have shape [batch_size, num_classes].")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    confidences, predictions = probabilities.max(dim=1)
    thresholded_predictions = predictions.clone()
    thresholded_predictions[confidences < threshold] = reject_label
    return thresholded_predictions, confidences
