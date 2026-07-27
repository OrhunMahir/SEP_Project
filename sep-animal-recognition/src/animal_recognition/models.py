"""Neural-network architectures used in the project."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    Swin_T_Weights,
    efficientnet_b0,
    resnet18,
    resnet50,
    swin_t,
)

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


class DoubleConvBNReLUPool(nn.Module):
    """Two 3x3 convolution layers followed by max-pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TripleConvBNReLUPool(nn.Module):
    """Three 3x3 convolution layers followed by max-pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DoubleConvKernelPool(nn.Module):
    """Two same-size convolution layers with a configurable odd kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention used in the SE ablation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.excitation(self.pool(inputs)).view(
            inputs.shape[0], inputs.shape[1], 1, 1
        )
        return inputs * weights


class DoubleConvSEPool(nn.Module):
    """Two convolution layers, SE attention, and max-pooling."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_channels, reduction),
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

        # Match the four-block 32-64-128-256 channel layout described in the report.
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


class CustomCNN8Conv(nn.Module):
    """Four-block, eight-convolution CNN trained entirely from scratch."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.3) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(f"CustomCNN8Conv requires {NUM_OUTPUTS} outputs, received {num_outputs}.")

        self.features = nn.Sequential(
            DoubleConvBNReLUPool(3, 32),
            DoubleConvBNReLUPool(32, 64),
            DoubleConvBNReLUPool(64, 128),
            DoubleConvBNReLUPool(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


class CustomCNN12Conv(nn.Module):
    """Four-block, twelve-convolution depth ablation."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.3) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"CustomCNN12Conv requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        self.features = nn.Sequential(
            TripleConvBNReLUPool(3, 32),
            TripleConvBNReLUPool(32, 64),
            TripleConvBNReLUPool(64, 128),
            TripleConvBNReLUPool(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


class CustomCNN8ConvKernel(nn.Module):
    """Eight-convolution kernel-size ablation."""

    def __init__(
        self,
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"CustomCNN8ConvKernel requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        self.features = nn.Sequential(
            DoubleConvKernelPool(3, 32, kernel_size),
            DoubleConvKernelPool(32, 64, kernel_size),
            DoubleConvKernelPool(64, 128, kernel_size),
            DoubleConvKernelPool(128, 256, kernel_size),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


class CustomCNN8ConvWide(nn.Module):
    """Eight-convolution width ablation."""

    def __init__(
        self,
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.3,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"CustomCNN8ConvWide requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain four positive integers.")
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            DoubleConvBNReLUPool(3, c1),
            DoubleConvBNReLUPool(c1, c2),
            DoubleConvBNReLUPool(c2, c3),
            DoubleConvBNReLUPool(c3, c4),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c4, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


class CustomCNN8ConvSE(nn.Module):
    """Eight-convolution squeeze-and-excitation ablation."""

    def __init__(
        self,
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.3,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"CustomCNN8ConvSE requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        self.features = nn.Sequential(
            DoubleConvSEPool(3, 32, reduction),
            DoubleConvSEPool(32, 64, reduction),
            DoubleConvSEPool(64, 128, reduction),
            DoubleConvSEPool(128, 256, reduction),
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


class ScratchResNet(nn.Module):
    """Configurable basic-block ResNet trained from random initialization."""

    def __init__(
        self,
        blocks: tuple[int, int, int, int],
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.0,
        model_name: str = "ScratchResNet",
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"{model_name} requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer_1 = self._make_layer(64, blocks=blocks[0], stride=1)
        self.layer_2 = self._make_layer(128, blocks=blocks[1], stride=2)
        self.layer_3 = self._make_layer(256, blocks=blocks[2], stride=2)
        self.layer_4 = self._make_layer(512, blocks=blocks[3], stride=2)
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


class ResNet18(ScratchResNet):
    """Standard ResNet-18 trained from random initialization for 21 outputs."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.0) -> None:
        super().__init__(
            blocks=(2, 2, 2, 2),
            num_outputs=num_outputs,
            dropout=dropout,
            model_name="ResNet18",
        )


class ResNet34(ScratchResNet):
    """Standard ResNet-34 depth ablation trained from random initialization."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.0) -> None:
        super().__init__(
            blocks=(3, 4, 6, 3),
            num_outputs=num_outputs,
            dropout=dropout,
            model_name="ResNet34",
        )


class ResNet50(nn.Module):
    """Standard ResNet-50 trained from random initialization for 21 outputs."""

    def __init__(self, num_outputs: int = NUM_OUTPUTS, dropout: float = 0.1) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(f"ResNet50 requires {NUM_OUTPUTS} outputs, received {num_outputs}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        self.network = resnet50(weights=None)
        self.network.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.network.fc.in_features, num_outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


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


class SwinTiny(nn.Module):
    """Swin-Tiny initialized from scratch without pretrained weights."""

    def __init__(
        self,
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(f"SwinTiny requires {NUM_OUTPUTS} outputs, received {num_outputs}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in the interval [0.0, 1.0).")
        self.network = swin_t(
            weights=None,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.network.head = nn.Linear(self.network.head.in_features, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_model(
    model_config: dict[str, object],
    *,
    initialize_pretrained: bool = True,
) -> nn.Module:
    """Build a supported model from an experiment configuration.

    ``initialize_pretrained`` should be disabled when a complete project
    checkpoint will immediately be loaded.  This preserves the architecture
    selected by ``pretrained`` without downloading redundant ImageNet weights.
    """
    model_name = str(model_config["name"])
    num_outputs = int(model_config["num_outputs"])
    dropout = float(model_config.get("dropout", 0.0))
    pretrained = bool(model_config.get("pretrained", False))
    if model_name == "custom_cnn":
        if pretrained:
            raise ValueError("CustomCNN does not support pretrained initialization.")
        return CustomCNN(num_outputs=num_outputs, dropout=dropout)
    if model_name == "custom_cnn_8conv":
        if pretrained:
            raise ValueError("CustomCNN8Conv does not support pretrained initialization.")
        return CustomCNN8Conv(num_outputs=num_outputs, dropout=dropout)
    if model_name == "custom_cnn_12conv":
        return CustomCNN12Conv(num_outputs=num_outputs, dropout=dropout)
    if model_name == "custom_cnn_8conv_k5":
        return CustomCNN8ConvKernel(
            num_outputs=num_outputs,
            dropout=dropout,
            kernel_size=int(model_config.get("kernel_size", 5)),
        )
    if model_name == "custom_cnn_8conv_wide":
        channels = tuple(int(value) for value in model_config.get("channels", (48, 96, 192, 384)))
        return CustomCNN8ConvWide(
            num_outputs=num_outputs,
            dropout=dropout,
            channels=channels,
        )
    if model_name == "custom_cnn_8conv_se":
        return CustomCNN8ConvSE(
            num_outputs=num_outputs,
            dropout=dropout,
            reduction=int(model_config.get("se_reduction", 16)),
        )
    if model_name == "resnet18":
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1 if initialize_pretrained else None
            model = resnet18(weights=weights)
            classifier: nn.Module = nn.Linear(model.fc.in_features, num_outputs)
            if dropout > 0.0:
                classifier = nn.Sequential(nn.Dropout(dropout), classifier)
            model.fc = classifier
            return model
        return ResNet18(num_outputs=num_outputs, dropout=dropout)
    if model_name == "resnet34":
        if pretrained:
            raise ValueError("The reported ResNet-34 ablation was trained from scratch.")
        return ResNet34(num_outputs=num_outputs, dropout=dropout)
    if model_name == "resnet50":
        if pretrained:
            raise ValueError("ResNet50 pretrained initialization is not configured.")
        return ResNet50(num_outputs=num_outputs, dropout=dropout)
    if model_name == "efficientnet_b0":
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if initialize_pretrained else None
            model = efficientnet_b0(weights=weights, dropout=dropout)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_outputs)
            return model
        return EfficientNetB0(num_outputs=num_outputs, dropout=dropout)
    if model_name == "swin_tiny":
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1 if initialize_pretrained else None
            model = swin_t(
                weights=weights,
                dropout=dropout,
                attention_dropout=float(model_config.get("attention_dropout", 0.0)),
            )
            model.head = nn.Linear(model.head.in_features, num_outputs)
            return model
        return SwinTiny(
            num_outputs=num_outputs,
            dropout=dropout,
            attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of model parameters that will be optimized."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
