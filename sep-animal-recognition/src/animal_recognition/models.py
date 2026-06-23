"""Neural-network architectures used in the project."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b0

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


class ResidualBlock(nn.Module):
    """The two-convolution basic residual block used by ResNet-18."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.convolution_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.activation(self.batch_norm_1(self.convolution_1(inputs)))
        features = self.batch_norm_2(self.convolution_2(features))
        return self.activation(features + residual)


class ResNet18(nn.Module):
    """Standard ResNet-18 trained from random initialization for 21 outputs."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.0) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(f"ResNet18 requires {NUM_OUTPUTS} outputs, received {num_outputs}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer_1 = self._make_layer(64, blocks=2, stride=1)
        self.layer_2 = self._make_layer(128, blocks=2, stride=2)
        self.layer_3 = self._make_layer(256, blocks=2, stride=2)
        self.layer_4 = self._make_layer(512, blocks=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_outputs)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(ResidualBlock(self.in_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.layer_1(features)
        features = self.layer_2(features)
        features = self.layer_3(features)
        features = self.layer_4(features)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 initialized from scratch without pretrained weights."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.2) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"EfficientNetB0 requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        self.network = efficientnet_b0(weights=None, dropout=dropout)
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_model(model_config: dict[str, object]) -> nn.Module:
    """Build a supported randomly initialized model from an experiment configuration."""
    model_name = str(model_config["name"])
    num_outputs = int(model_config["num_outputs"])
    dropout = float(model_config.get("dropout", 0.0))
    if model_name == "custom_cnn":
        return CustomCNN(num_outputs=num_outputs, dropout=dropout)
    if model_name == "resnet18":
        return ResNet18(num_outputs=num_outputs, dropout=dropout)
    if model_name == "efficientnet_b0":
        return EfficientNetB0(num_outputs=num_outputs, dropout=dropout)
    raise ValueError(f"Unsupported model name: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of model parameters that will be optimized."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
