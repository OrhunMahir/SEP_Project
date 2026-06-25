#!/usr/bin/env python3
"""Calibrate the twelve-convolution custom CNN with the shared threshold sweep."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_custom_cnn_12conv import CustomCNN12Conv


def build_custom_cnn_12conv(model_config: dict[str, object]) -> CustomCNN12Conv:
    """Build the architecture matching the saved twelve-convolution checkpoint."""
    return CustomCNN12Conv(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
    )


if __name__ == "__main__":
    calibration.build_model = build_custom_cnn_12conv
    calibration.main()
