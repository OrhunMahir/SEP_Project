#!/usr/bin/env python3
"""Train a scratch model and select the best checkpoint by validation macro-F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from animal_recognition.data import (
    BalancedEpochSampler,
    ManifestDataset,
    evaluation_transform,
    load_split,
    training_transform,
)
from animal_recognition.models import build_model, count_trainable_parameters
from animal_recognition.training import (
    evaluate_model,
    set_seed,
    set_warmup_cosine_learning_rate,
    train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "custom_cnn_baseline.json",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
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
    num_workers = args.num_workers if args.num_workers is not None else int(data_config["num_workers"])
    image_root = resolve_project_path(str(data_config.get("image_root", data_paths["train_image_root"])))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Configured image root was not found: {image_root}")
    train_samples = load_split(resolve_project_path(data_config["train_split"]), image_root)
    validation_samples = load_split(resolve_project_path(data_config["validation_split"]), image_root)

    train_dataset = ManifestDataset(
        train_samples,
        training_transform(
            int(data_config["image_size"]),
            config.get("augmentation"),
        ),
    )
    validation_dataset = ManifestDataset(
        validation_samples, evaluation_transform(int(data_config["image_size"]))
    )
    sampler = BalancedEpochSampler(train_samples, seed=int(config["seed"]))

    # Reject sınıfını dengelemek için sadece train loader'da balanced sampler kullanıyorum.
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

    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_config["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_macro_f1 = float("-inf")
    best_validation_metrics: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history: list[dict] = []
    started_at = time.perf_counter()

    print(f"Device: {device}")
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
        record = {
            "epoch": epoch_index + 1,
            "learning_rate": learning_rate,
            "train_loss": train_result["loss"],
            **{f"validation_{key}": value for key, value in validation_result.items()},
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        macro_f1 = float(validation_result["macro_f1"])
        print(
            f"Epoch {epoch_index + 1:02d}/{max_epochs} | "
            f"train_loss={train_result['loss']:.4f} | "
            f"val_loss={validation_result['loss']:.4f} | "
            f"accuracy={validation_result['accuracy']:.4f} | "
            f"lr={learning_rate:.6f}\n"
            f"  macro(P/R/F1)={validation_result['macro_precision']:.4f}/"
            f"{validation_result['macro_recall']:.4f}/{macro_f1:.4f} | "
            f"weighted_f1={validation_result['weighted_f1']:.4f}\n"
            f"  reject(P/R/F1)={validation_result['reject_precision']:.4f}/"
            f"{validation_result['reject_recall']:.4f}/"
            f"{validation_result['reject_f1']:.4f} | "
            f"false_accepts={validation_result['false_accepts']} | "
            f"false_rejects={validation_result['false_rejects']}"
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_validation_metrics = dict(validation_result)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": config["model"]["name"],
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
            print("Early stopping: macro-F1 did not improve within the configured patience.")
            break

    summary = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "best_validation_macro_f1": best_macro_f1,
        "best_validation_metrics": best_validation_metrics,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
