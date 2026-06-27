#!/usr/bin/env python3
"""Calibrate the eight-convolution SE-CNN with the shared threshold sweep."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_custom_cnn_8conv_se import CustomCNN8ConvSE


def build_custom_cnn_8conv_se(model_config: dict[str, object]) -> CustomCNN8ConvSE:
    """Build the architecture matching the saved SE-CNN checkpoint."""
    return CustomCNN8ConvSE(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
        se_reduction=int(model_config.get("se_reduction", 16)),
    )


if __name__ == "__main__":
    calibration.build_model = build_custom_cnn_8conv_se
    calibration.main()
