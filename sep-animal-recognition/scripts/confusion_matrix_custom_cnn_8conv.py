#!/usr/bin/env python3
"""Create validation confusion matrices for the selected 8-conv custom CNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from animal_recognition.constants import CLASSES, NUM_OUTPUTS, REJECT_INTERNAL
from animal_recognition.data import ManifestDataset, evaluation_transform, load_split
from animal_recognition.thresholding import apply_confidence_threshold
from train_custom_cnn_8conv import CustomCNN8Conv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [*CLASSES, "reject"]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def write_matrix_csv(path: Path, matrix: list[list[int | float]]) -> None:
    """Write a labelled confusion matrix to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix, strict=True):
            writer.writerow([class_name, *row])


def write_predictions_csv(
    path: Path,
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    relative_paths: list[str],
) -> None:
    """Write per-image validation predictions for error inspection."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "true_label",
                "predicted_label",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for target, prediction, confidence, relative_path in zip(
            targets, predictions, confidences, relative_paths, strict=True
        ):
            writer.writerow({
                "filename": relative_path,
                "true_label": CLASS_NAMES[target],
                "predicted_label": CLASS_NAMES[prediction],
                "confidence": f"{confidence:.6f}",
                "correct": int(target == prediction),
            })


def write_png(path: Path, matrix: list[list[int]], *, normalize: bool) -> None:
    """Write a confusion matrix heatmap with matplotlib or a Pillow fallback."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        write_png_with_pillow(path, matrix, normalize=normalize)
        return

    display_matrix = np.asarray(matrix)
    values_format = ".2f" if normalize else "d"
    title = "Normalized confusion matrix" if normalize else "Confusion matrix"
    if normalize:
        display_matrix = display_matrix.astype(float)
        row_totals = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix, dtype=float),
            where=row_totals != 0,
        )

    figure, axis = plt.subplots(figsize=(16, 16))
    display = ConfusionMatrixDisplay(
        confusion_matrix=display_matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=axis,
        xticks_rotation=90,
        values_format=values_format,
        colorbar=True,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_png_with_pillow(path: Path, matrix: list[list[int]], *, normalize: bool) -> None:
    """Write a compact heatmap using Pillow when matplotlib is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Neither matplotlib nor Pillow is available; skipping PNG output.")
        return

    display_matrix: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        if normalize:
            display_matrix.append([
                value / row_total if row_total else 0.0 for value in row
            ])
        else:
            display_matrix.append([float(value) for value in row])

    max_value = max(max(row) for row in display_matrix) if display_matrix else 0.0
    max_value = max(max_value, 1e-12)

    cell_size = 44
    left_margin = 230
    top_margin = 260
    right_margin = 40
    bottom_margin = 40
    width = left_margin + cell_size * NUM_OUTPUTS + right_margin
    height = top_margin + cell_size * NUM_OUTPUTS + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = "Row-normalized confusion matrix" if normalize else "Confusion matrix counts"
    draw.text((left_margin, 20), title, fill="black", font=font)
    draw.text((left_margin, 42), "Predicted label", fill="black", font=font)
    draw.text((10, top_margin - 25), "True label", fill="black", font=font)

    for index, class_name in enumerate(CLASS_NAMES):
        x = left_margin + index * cell_size + 4
        y = top_margin - 8
        draw.text((x, y), class_name[:12], fill="black", font=font, anchor="ls")
        label_image = Image.new("RGBA", (120, 18), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((0, 0), class_name[:18], fill="black", font=font)
        label_image = label_image.rotate(60, expand=True)
        image.paste(label_image, (x - 10, top_margin - 150), label_image)

        row_y = top_margin + index * cell_size + cell_size // 2
        draw.text((10, row_y), class_name, fill="black", font=font, anchor="lm")

    for row_index, row in enumerate(display_matrix):
        for column_index, value in enumerate(row):
            intensity = value / max_value
            red = int(255 - 215 * intensity)
            green = int(255 - 225 * intensity)
            blue = 255
            x0 = left_margin + column_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=(red, green, blue), outline=(210, 210, 210))
            label = f"{value:.2f}" if normalize else str(int(value))
            text_fill = "white" if intensity > 0.55 else "black"
            draw.text(
                (x0 + cell_size / 2, y0 + cell_size / 2),
                label,
                fill=text_fill,
                font=font,
                anchor="mm",
            )

    image.save(path)
    print(f"matplotlib is not available; wrote Pillow fallback PNG: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "custom_cnn_8conv_medium_aug.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
    model_config = config["model"]
    if model_config["name"] != "custom_cnn_8conv":
        raise ValueError("This script only supports model.name='custom_cnn_8conv'.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    output_dir = resolve_project_path(str(config["output_dir"]))
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

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
    validation_loader = DataLoader(
        ManifestDataset(
            validation_samples,
            evaluation_transform(int(data_config["image_size"])),
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
        threshold=float(args.threshold),
    )

    labels = list(range(NUM_OUTPUTS))
    matrix = confusion_matrix(targets, predictions, labels=labels).tolist()
    normalized_matrix: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        normalized_matrix.append([
            round(value / row_total, 6) if row_total else 0.0 for value in row
        ])

    threshold_label = f"{args.threshold:.2f}".replace(".", "p")
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

    correct = sum(target == prediction for target, prediction in zip(targets, predictions, strict=True))
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
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Validation samples: {len(targets)}")
    print(f"Accuracy from confusion matrix: {correct / len(targets):.4f}")
    print(f"False accepts: {false_accepts}")
    print(f"False rejects: {false_rejects}")
    print(f"Saved confusion matrix files to: {confusion_dir}")


if __name__ == "__main__":
    main()
