#!/usr/bin/env python3
"""Calibrate the eight-convolution custom CNN with 5x5 kernels."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_custom_cnn_8conv_k5 import CustomCNN8ConvK5


def build_custom_cnn_8conv_k5(model_config: dict[str, object]) -> CustomCNN8ConvK5:
    """Build the architecture matching the saved 5x5-kernel checkpoint."""
    return CustomCNN8ConvK5(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
        kernel_size=int(model_config.get("kernel_size", 5)),
    )


if __name__ == "__main__":
    calibration.build_model = build_custom_cnn_8conv_k5
    calibration.main()
