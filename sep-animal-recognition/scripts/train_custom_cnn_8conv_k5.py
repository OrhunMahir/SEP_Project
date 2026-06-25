#!/usr/bin/env python3
"""Train an eight-convolution custom CNN using wider 5x5 kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from animal_recognition.constants import NUM_OUTPUTS
from animal_recognition.data import (
    BalancedEpochSampler,
    ManifestDataset,
    evaluation_transform,
    load_split,
    training_transform,
)
from animal_recognition.models import count_trainable_parameters
from animal_recognition.training import (
    evaluate_model,
    set_seed,
    set_warmup_cosine_learning_rate,
    train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TwoConvBlockK5(nn.Module):
    """Two Conv-BN-ReLU layers with configurable odd kernels, then downsampling."""

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


class CustomCNN8ConvK5(nn.Module):
    """Eight-convolution CNN with four two-convolution feature blocks."""

    def __init__(
        self,
        num_outputs: int = NUM_OUTPUTS,
        dropout: float = 0.3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if num_outputs != NUM_OUTPUTS:
            raise ValueError(
                f"CustomCNN8ConvK5 requires {NUM_OUTPUTS} outputs, received {num_outputs}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        self.features = nn.Sequential(
            TwoConvBlockK5(3, 32, kernel_size),
            TwoConvBlockK5(32, 64, kernel_size),
            TwoConvBlockK5(64, 128, kernel_size),
            TwoConvBlockK5(128, 256, kernel_size),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled))


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "custom_cnn_8conv_k5.json",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
    model_config = config["model"]
    if model_config["name"] != "custom_cnn_8conv_k5":
        raise ValueError("This script only supports model.name='custom_cnn_8conv_k5'.")

    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    set_seed(int(config["seed"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    data_config = config["data"]
    training_config = config["training"]
    max_epochs = args.max_epochs or int(training_config["max_epochs"])
    selection_metric = str(training_config.get("selection_metric", "macro_f1"))
    if selection_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("selection_metric must be either 'accuracy' or 'macro_f1'.")

    image_root = resolve_project_path(str(data_paths["train_image_root"]))
    train_samples = load_split(
        resolve_project_path(str(data_config["train_split"])), image_root
    )
    validation_samples = load_split(
        resolve_project_path(str(data_config["validation_split"])), image_root
    )
    image_size = int(data_config["image_size"])
    train_dataset = ManifestDataset(train_samples, training_transform(image_size))
    validation_dataset = ManifestDataset(
        validation_samples, evaluation_transform(image_size)
    )
    sampler = BalancedEpochSampler(train_samples, seed=int(config["seed"]))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_config["num_workers"])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    kernel_size = int(model_config.get("kernel_size", 5))
    model = CustomCNN8ConvK5(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
        kernel_size=kernel_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    output_dir = resolve_project_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_selection_score = float("-inf")
    best_validation_metrics: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    print(f"Device: {device}")
    print(
        "Architecture: 8 convolution layers "
        f"(2 per feature block, {kernel_size}x{kernel_size} kernels)"
    )
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    for epoch_index in range(max_epochs):
        sampler.set_epoch(epoch_index)
        learning_rate = set_warmup_cosine_learning_rate(
            optimizer=optimizer,
            base_learning_rate=float(training_config["learning_rate"]),
            epoch_index=epoch_index,
            warmup_epochs=int(training_config["warmup_epochs"]),
            max_epochs=max_epochs,
        )
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_result = evaluate_model(model, validation_loader, criterion, device)
        selection_score = float(validation_result[selection_metric])
        record: dict[str, float | int] = {
            "epoch": epoch_index + 1,
            "learning_rate": learning_rate,
            "train_loss": train_result["loss"],
            **{f"validation_{key}": value for key, value in validation_result.items()},
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(
            f"Epoch {epoch_index + 1:02d}/{max_epochs} | "
            f"train_loss={train_result['loss']:.4f} | "
            f"val_loss={validation_result['loss']:.4f} | "
            f"accuracy={validation_result['accuracy']:.4f} | "
            f"macro_f1={validation_result['macro_f1']:.4f} | "
            f"reject_f1={validation_result['reject_f1']:.4f} | "
            f"lr={learning_rate:.6f}"
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_validation_metrics = dict(validation_result)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": "custom_cnn_8conv_k5",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch_index + 1,
                    "validation_metrics": validation_result,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
            print(
                f"Early stopping: {selection_metric} did not improve within "
                "the configured patience."
            )
            break

    summary = {
        "experiment_name": config["experiment_name"],
        "architecture": "custom_cnn_8conv_k5",
        "kernel_size": kernel_size,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "selected_checkpoint_metrics": best_validation_metrics,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
