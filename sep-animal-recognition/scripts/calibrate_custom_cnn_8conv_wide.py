#!/usr/bin/env python3
"""Calibrate the wider eight-convolution custom CNN with the shared threshold sweep."""

from __future__ import annotations

import calibrate_threshold as calibration

from train_custom_cnn_8conv_wide import CustomCNN8ConvWide


def build_custom_cnn_8conv_wide(model_config: dict[str, object]) -> CustomCNN8ConvWide:
    """Build the architecture matching the saved wider 8-conv checkpoint."""
    return CustomCNN8ConvWide(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
        channels=[int(channel) for channel in model_config.get("channels", [48, 96, 192, 384])],
    )


if __name__ == "__main__":
    calibration.build_model = build_custom_cnn_8conv_wide
    calibration.main()
