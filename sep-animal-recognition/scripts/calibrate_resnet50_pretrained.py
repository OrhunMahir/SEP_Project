#!/usr/bin/env python3
"""Calibrate a pretrained ResNet-50 checkpoint with the shared threshold sweep."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_resnet50_pretrained import PretrainedResNet50


def build_resnet50(model_config: dict[str, object]) -> PretrainedResNet50:
    """Build the checkpoint-compatible ResNet-50 used by this experiment."""
    return PretrainedResNet50(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )


if __name__ == "__main__":
    calibration.build_model = build_resnet50
    calibration.main()
