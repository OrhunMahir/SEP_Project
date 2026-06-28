"""Reusable training, evaluation, seeding, and learning-rate utilities."""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_warmup_cosine_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    epoch_index: int,
    warmup_epochs: int,
    max_epochs: int,
) -> float:
    """Apply linear warm-up followed by cosine decay and return the active rate."""
    # İlk birkaç epoch'ta öğrenme oranını yavaşça artırıp eğitimi daha kararlı başlatıyorum.
    if epoch_index < warmup_epochs:
        multiplier = (epoch_index + 1) / warmup_epochs
    else:
        progress = (epoch_index - warmup_epochs) / max(1, max_epochs - warmup_epochs - 1)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    learning_rate = base_learning_rate * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one optimizer pass over the training loader."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_examples += images.size(0)
    return {"loss": total_loss / max(total_examples, 1)}


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a model with deterministic validation data and full metrics."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        total_examples += images.size(0)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result = classification_metrics(targets, predictions)
    result["loss"] = total_loss / max(total_examples, 1)
    return result




@torch.inference_mode()
def predict_model(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]],
    device: torch.device,
) -> tuple[list[int], list[int], list[str]]:
    """Collect deterministic model predictions, targets, and relative image paths."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    relative_paths: list[str] = []
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        relative_paths.extend(paths)
    return targets, predictions, relative_paths
