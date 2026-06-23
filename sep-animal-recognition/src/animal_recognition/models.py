"""Neural-network architectures used in the project."""

from __future__ import annotations

import torch
from torch import nn

from .constants import NUM_OUTPUTS


class ConvBNReLUPool(nn.Module):
    """One convolutional feature-extraction block from the report architecture."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CustomCNN(nn.Module):
    """Four-block CNN baseline trained entirely from random initialization."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.3) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(f"CustomCNN requires {NUM_OUTPUTS} outputs, received {num_outputs}.")

        # Bu dosyada rapordaki 32→64→128→256 kanallı dört bloğu aynen kuruyorum.
        self.features = nn.Sequential(
            ConvBNReLUPool(3, 32),
            ConvBNReLUPool(32, 64),
            ConvBNReLUPool(64, 128),
            ConvBNReLUPool(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of model parameters that will be optimized."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
