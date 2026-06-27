"""Grad-CAM utilities adapted to torchvision Swin feature layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GradCAMResult:
    """One Grad-CAM explanation and the classifier outputs that produced it."""

    heatmap: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    target_class: int


def unwrap_swin(model: nn.Module) -> nn.Module:
    """Return the torchvision Swin module from a project wrapper when necessary."""
    network = getattr(model, "network", None)
    return network if isinstance(network, nn.Module) else model


def resolve_swin_target_layer(
    model: nn.Module,
    target_layer: str = "last_block_norm1",
) -> nn.Module:
    """Resolve a supported channels-last Swin layer for Grad-CAM."""
    swin = unwrap_swin(model)
    if target_layer == "final_norm":
        layer = getattr(swin, "norm", None)
    elif target_layer in {"last_block_norm1", "last_block_norm2"}:
        try:
            last_block = swin.features[-1][-1]
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "Could not locate the final Swin transformer block."
            ) from error
        layer = getattr(last_block, target_layer.removeprefix("last_block_"), None)
    else:
        raise ValueError(
            "Unsupported target layer. Choose last_block_norm1, "
            "last_block_norm2, or final_norm."
        )

    if not isinstance(layer, nn.Module):
        raise ValueError(f"Could not resolve Swin target layer: {target_layer}")
    return layer


class SwinGradCAM:
    """Generate positive-evidence Grad-CAM maps from channels-last Swin activations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self._activations: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError(
                "The selected Swin target layer must return a four-dimensional "
                "channels-last tensor [batch, height, width, channels]."
            )
        self._activations = output

    def close(self) -> None:
        """Remove the forward hook owned by this explainer."""
        self._hook.remove()

    def __enter__(self) -> SwinGradCAM:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """Run one forward/backward pass and return a normalized 2D heatmap."""
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently requires one image with shape [1, C, H, W]."
            )

        self._activations = None
        self.model.zero_grad(set_to_none=True)

        model_input = input_tensor.detach().requires_grad_(True)
        logits = self.model(model_input)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                "The classifier must return logits with shape [1, num_classes]."
            )
        if self._activations is None:
            raise RuntimeError(
                "The target layer did not run during the model forward pass."
            )

        predicted_class = int(logits.argmax(dim=1).item())
        selected_class = predicted_class if target_class is None else int(target_class)
        if not 0 <= selected_class < logits.shape[1]:
            raise ValueError(
                f"Target class {selected_class} is outside [0, {logits.shape[1] - 1}]."
            )

        gradients = torch.autograd.grad(
            outputs=logits[0, selected_class],
            inputs=self._activations,
            retain_graph=False,
            create_graph=False,
        )[0]
        activations = self._activations
        if activations.shape != gradients.shape:
            raise RuntimeError("Activation and gradient shapes differ unexpectedly.")

        channel_weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.relu((channel_weights * activations).sum(dim=-1))[0]
        maximum = heatmap.max()
        if torch.isfinite(maximum) and float(maximum.detach().item()) > 0.0:
            heatmap = heatmap / maximum
        else:
            heatmap = torch.zeros_like(heatmap)

        return GradCAMResult(
            heatmap=heatmap.detach().cpu(),
            probabilities=torch.softmax(logits.detach(), dim=1)[0].cpu(),
            predicted_class=predicted_class,
            target_class=selected_class,
        )