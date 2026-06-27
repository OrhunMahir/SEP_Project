#!/usr/bin/env python3
"""Create validation confusion matrices for the YOLO-cropped 8-conv CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from animal_recognition.constants import NUM_OUTPUTS, REJECT_INTERNAL
from animal_recognition.data import evaluation_transform, load_split
from animal_recognition.thresholding import apply_confidence_threshold
from train_custom_cnn_8conv import CustomCNN8Conv
from train_custom_cnn_8conv_yolo import (
    PROJECT_ROOT,
    YoloCropDataset,
    load_or_create_yolo_boxes,
    read_json,
    resolve_project_path,
)
from confusion_matrix_custom_cnn_8conv import (
    CLASS_NAMES,
    write_matrix_csv,
    write_png,
    write_predictions_csv,
)


def read_best_threshold(output_dir: Path) -> float:
    """Read the macro-F1-selected threshold from the calibration summary."""
    summary_path = output_dir / "threshold_calibration" / "threshold_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Threshold summary was not found. Run calibration first or pass --threshold: "
            f"{summary_path}"
        )
    summary = read_json(summary_path)
    return float(summary["best_threshold"]["threshold"])


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[list[int], list[int], list[float], list[str]]:
    """Collect thresholded predictions and confidence scores on validation data."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    relative_paths: list[str] = []

    with torch.no_grad():
        for images, batch_targets, batch_paths in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1).cpu()
            batch_predictions, batch_confidences = apply_confidence_threshold(
                probabilities, threshold
            )
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.tolist())
            confidences.extend(float(value) for value in batch_confidences.tolist())
            relative_paths.extend(batch_paths)

    return targets, predictions, confidences, relative_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "custom_cnn_8conv_yolo_medium_aug.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
    model_config = config["model"]
    if model_config["name"] != "custom_cnn_8conv_yolo":
        raise ValueError("This script only supports model.name='custom_cnn_8conv_yolo'.")

    output_dir = resolve_project_path(str(config["output_dir"]))
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    threshold = read_best_threshold(output_dir) if args.threshold is None else float(args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    model = CustomCNN8Conv(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    data_config = checkpoint_config["data"]
    validation_samples = load_split(
        resolve_project_path(str(data_config["validation_split"])),
        resolve_project_path(str(data_paths["train_image_root"])),
    )
    crop_records = load_or_create_yolo_boxes(
        samples=validation_samples,
        config=checkpoint_config,
        output_dir=output_dir,
        device=device,
        force=False,
    )
    validation_loader = DataLoader(
        YoloCropDataset(
            validation_samples,
            evaluation_transform(int(data_config["image_size"])),
            crop_records,
        ),
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    targets, predictions, confidences, relative_paths = collect_predictions(
        model=model,
        loader=validation_loader,
        device=device,
        threshold=threshold,
    )

    matrix = confusion_matrix(targets, predictions, labels=list(range(NUM_OUTPUTS))).tolist()
    normalized_matrix: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        normalized_matrix.append([
            round(value / row_total, 6) if row_total else 0.0 for value in row
        ])

    threshold_label = f"{threshold:.2f}".replace(".", "p")
    confusion_dir = args.output_dir or output_dir / f"confusion_matrix_t{threshold_label}"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    write_matrix_csv(confusion_dir / "confusion_matrix_counts.csv", matrix)
    write_matrix_csv(confusion_dir / "confusion_matrix_row_normalized.csv", normalized_matrix)
    write_predictions_csv(
        confusion_dir / "validation_predictions.csv",
        targets,
        predictions,
        confidences,
        relative_paths,
    )
    write_png(confusion_dir / "confusion_matrix_counts.png", matrix, normalize=False)
    write_png(confusion_dir / "confusion_matrix_row_normalized.png", matrix, normalize=True)

    correct = sum(
        target == prediction for target, prediction in zip(targets, predictions, strict=True)
    )
    false_accepts = sum(
        target == REJECT_INTERNAL and prediction != REJECT_INTERNAL
        for target, prediction in zip(targets, predictions, strict=True)
    )
    false_rejects = sum(
        target != REJECT_INTERNAL and prediction == REJECT_INTERNAL
        for target, prediction in zip(targets, predictions, strict=True)
    )
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Validation samples: {len(targets)}")
    print(f"Accuracy from confusion matrix: {correct / len(targets):.4f}")
    print(f"False accepts: {false_accepts}")
    print(f"False rejects: {false_rejects}")
    print(f"Saved confusion matrix files to: {confusion_dir}")


if __name__ == "__main__":
    main()
