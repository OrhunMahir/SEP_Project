#!/usr/bin/env python3
"""Run YOLO-gated Swin-Tiny inference on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torchvision import transforms
from torchvision.models import Swin_T_Weights, swin_t

from animal_recognition.constants import (
    CLASSES,
    NUM_OUTPUTS,
    REJECT_EXTERNAL,
    REJECT_INTERNAL,
    external_to_internal,
    internal_to_external,
)
from animal_recognition.yolo_crop import (
    Detection,
    padded_square_crop_box,
    select_largest_target_animal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFUSION_EXTERNAL_LABELS = [REJECT_EXTERNAL, *range(len(CLASSES))]
CONFUSION_INTERNAL_LABELS = [REJECT_INTERNAL, *range(len(CLASSES))]
CONFUSION_NAMES = ["reject(-1)", *CLASSES]


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path: Path | str) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def get_nested(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """Read a nested config value without requiring one exact JSON shape."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_swin_tiny(model_config: dict[str, Any], load_from_checkpoint: bool) -> nn.Module:
    """Build a 21-output Swin-Tiny classifier for ImageNet pretraining or checkpoint loading."""
    num_outputs = int(model_config.get("num_outputs", NUM_OUTPUTS))
    if num_outputs != NUM_OUTPUTS:
        raise ValueError(f"Swin-Tiny requires {NUM_OUTPUTS} outputs, received {num_outputs}.")

    dropout = float(model_config.get("dropout", 0.1))
    attention_dropout = float(model_config.get("attention_dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0.0, 1.0).")
    if not 0.0 <= attention_dropout < 1.0:
        raise ValueError("attention_dropout must be in the interval [0.0, 1.0).")

    use_imagenet_weights = bool(model_config.get("pretrained", True)) and not load_from_checkpoint
    model = swin_t(
        weights=Swin_T_Weights.IMAGENET1K_V1 if use_imagenet_weights else None,
        dropout=dropout,
        attention_dropout=attention_dropout,
    )
    model.head = nn.Linear(model.head.in_features, NUM_OUTPUTS)
    return model


def inference_transform(image_size: int) -> transforms.Compose:
    """Create the deterministic validation/inference transform."""
    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read validation rows with filename and external label."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"filename", "label"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns: {sorted(required_columns)}")
        return [{"filename": row["filename"], "label": row["label"]} for row in reader]


def detections_from_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics result into project-neutral detections."""
    if result.boxes is None:
        return []

    names = result.names
    detections: list[Detection] = []
    for class_id, confidence, coordinates in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        class_index = int(class_id)
        class_name = names[class_index] if isinstance(names, dict) else names[class_index]
        detections.append(
            Detection(
                class_name=str(class_name),
                confidence=float(confidence),
                x1=float(coordinates[0]),
                y1=float(coordinates[1]),
                x2=float(coordinates[2]),
                y2=float(coordinates[3]),
            )
        )
    return detections


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint and normalize the expected keys."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {path}")
    return checkpoint


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one prediction row per successfully processed validation image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "true_label",
                "predicted_label",
                "confidence",
                "yolo_detected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    """Write a 21x21 confusion matrix with stable reject-first labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *CONFUSION_NAMES])
        for label, row in zip(CONFUSION_NAMES, matrix, strict=True):
            writer.writerow([label, *row])


def write_confusion_plot(path: Path, matrix: list[list[int]], title: str) -> None:
    """Render a high-resolution count-annotated confusion-matrix heatmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=250)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Count")
    colorbar.mappable.set_clim(0, 250)

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xticks(range(NUM_OUTPUTS))
    ax.set_yticks(range(NUM_OUTPUTS))
    ax.set_xticklabels(CONFUSION_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(CONFUSION_NAMES, fontsize=8)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            text_color = "white" if value >= 125 else "black"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits" / "val_seed42.csv")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset" / "all")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PROJECT_ROOT / "runs" / "swin_tiny_pretrained_yolo" / "yolo_swin_predictions.csv",
    )
    parser.add_argument(
        "--confusion-csv",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "swin_tiny_pretrained_yolo"
        / "yolo_swin_confusion_matrix.csv",
    )
    parser.add_argument(
        "--confusion-plot",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "swin_tiny_pretrained_yolo"
        / "yolo_swin_confusion_matrix.png",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_json(resolve_project_path(args.config))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = load_checkpoint(resolve_project_path(args.checkpoint), device)
    checkpoint_config = checkpoint.get("config")
    model_config = dict(config.get("model", {}))
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_config = dict(checkpoint_config["model"])
    if model_config.get("name", "swin_tiny") != "swin_tiny":
        raise ValueError(f"This script expects a swin_tiny model config, got: {model_config.get('name')}")

    model = build_swin_tiny(model_config, load_from_checkpoint=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(get_nested(config, ("data", "image_size"), 224))
    swin_threshold = float(
        config.get(
            "threshold",
            get_nested(config, ("postprocessing", "threshold"), get_nested(config, ("thresholding", "threshold"), 0.0)),
        )
    )
    yolo_model_name = str(get_nested(config, ("yolo", "model"), "yolov8n.pt"))
    yolo_confidence = float(get_nested(config, ("yolo", "confidence_threshold"), 0.25))
    padding_fraction = float(get_nested(config, ("yolo", "padding_fraction"), 0.10))

    if not 0.0 <= swin_threshold <= 1.0:
        raise ValueError("Swin postprocessing threshold must be in the interval [0.0, 1.0].")
    if not 0.0 <= yolo_confidence <= 1.0:
        raise ValueError("YOLO confidence threshold must be in the interval [0.0, 1.0].")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for YOLO-gated Swin inference. "
            "Install requirements-yolo.txt first."
        ) from error

    transform = inference_transform(image_size)
    detector = YOLO(yolo_model_name)
    rows = read_manifest(resolve_project_path(args.manifest))
    dataset_root = resolve_project_path(args.dataset_root)

    prediction_rows: list[dict[str, Any]] = []
    true_internal_labels: list[int] = []
    predicted_internal_labels: list[int] = []

    for row in rows:
        filename = row["filename"]
        image_path = dataset_root / filename
        if not image_path.is_file():
            warnings.warn(f"Skipping missing image: {image_path}", stacklevel=1)
            continue

        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            warnings.warn(f"Skipping unreadable image: {image_path} ({error})", stacklevel=1)
            continue

        true_external = int(row["label"])
        true_internal = external_to_internal(true_external)
        result = detector.predict(
            source=str(image_path),
            conf=yolo_confidence,
            device=args.yolo_device,
            verbose=False,
        )[0]
        detection = select_largest_target_animal(detections_from_result(result))

        if detection is None:
            predicted_external = REJECT_EXTERNAL
            predicted_internal = REJECT_INTERNAL
            confidence = 0.0
            yolo_detected = False
        else:
            crop_box = padded_square_crop_box(
                detection,
                image_width=image.width,
                image_height=image.height,
                padding_fraction=padding_fraction,
            )
            if crop_box is None:
                predicted_external = REJECT_EXTERNAL
                predicted_internal = REJECT_INTERNAL
                confidence = 0.0
                yolo_detected = False
            else:
                cropped_image = image.crop(crop_box)
                tensor = transform(cropped_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence_tensor, prediction_tensor = torch.max(probabilities, dim=1)
                confidence = float(confidence_tensor.item())
                raw_internal_prediction = int(prediction_tensor.item())
                if confidence < swin_threshold:
                    predicted_internal = REJECT_INTERNAL
                else:
                    predicted_internal = raw_internal_prediction
                predicted_external = internal_to_external(predicted_internal)
                yolo_detected = True

        true_internal_labels.append(true_internal)
        predicted_internal_labels.append(predicted_internal)
        prediction_rows.append(
            {
                "filename": filename,
                "true_label": true_external,
                "predicted_label": predicted_external,
                "confidence": f"{confidence:.6f}",
                "yolo_detected": yolo_detected,
            }
        )

    if not true_internal_labels:
        raise RuntimeError("No validation images were processed; no outputs were written.")

    matrix = confusion_matrix(
        true_internal_labels,
        predicted_internal_labels,
        labels=CONFUSION_INTERNAL_LABELS,
    ).astype(int)
    matrix_list = matrix.tolist()

    predictions_csv = resolve_project_path(args.predictions_csv)
    confusion_csv = resolve_project_path(args.confusion_csv)
    confusion_plot = resolve_project_path(args.confusion_plot)
    write_predictions_csv(predictions_csv, prediction_rows)
    write_confusion_csv(confusion_csv, matrix_list)
    write_confusion_plot(
        confusion_plot,
        matrix_list,
        f"YOLO-gated Swin-Tiny pretrained, tau={swin_threshold:.2f}",
    )

    accuracy = accuracy_score(true_internal_labels, predicted_internal_labels)
    macro_f1 = f1_score(
        true_internal_labels,
        predicted_internal_labels,
        labels=list(range(NUM_OUTPUTS)),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        true_internal_labels,
        predicted_internal_labels,
        labels=list(range(NUM_OUTPUTS)),
        average="weighted",
        zero_division=0,
    )
    reject_f1 = f1_score(
        true_internal_labels,
        predicted_internal_labels,
        labels=[REJECT_INTERNAL],
        average=None,
        zero_division=0,
    )[0]
    false_accepts = sum(
        true == REJECT_INTERNAL and predicted != REJECT_INTERNAL
        for true, predicted in zip(true_internal_labels, predicted_internal_labels, strict=True)
    )
    false_rejects = sum(
        true != REJECT_INTERNAL and predicted == REJECT_INTERNAL
        for true, predicted in zip(true_internal_labels, predicted_internal_labels, strict=True)
    )

    print("Confusion matrix rows=true, columns=predicted:")
    print(matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Reject-F1: {reject_f1:.4f}")
    print(f"False accepts: {false_accepts}")
    print(f"False rejects: {false_rejects}")
    print(
        classification_report(
            true_internal_labels,
            predicted_internal_labels,
            labels=CONFUSION_INTERNAL_LABELS,
            target_names=CONFUSION_NAMES,
            zero_division=0,
        )
    )
    print(f"Predictions CSV: {predictions_csv.resolve()}")
    print(f"Confusion matrix CSV: {confusion_csv.resolve()}")
    print(f"Confusion matrix PNG: {confusion_plot.resolve()}")


if __name__ == "__main__":
    main()
