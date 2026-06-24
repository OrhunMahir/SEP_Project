#!/usr/bin/env python3
"""Calibrate the eight-convolution custom CNN with the shared threshold sweep."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_custom_cnn_8conv import CustomCNN8Conv


def build_custom_cnn_8conv(model_config: dict[str, object]) -> CustomCNN8Conv:
    """Build the architecture matching the saved eight-convolution checkpoint."""
    return CustomCNN8Conv(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
    )


if __name__ == "__main__":
    calibration.build_model = build_custom_cnn_8conv
    calibration.main()
