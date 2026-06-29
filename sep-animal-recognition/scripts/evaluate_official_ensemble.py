#!/usr/bin/env python3
"""Evaluate a fixed weighted ensemble on the course image-folder interface.

This is meant for the official validation/test folder used by the provided
inference.py contract: a flat image directory with an optional labels.csv file.
If labels.csv is present, metrics are written; otherwise only predictions and
timing are produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    config_path: Path
    checkpoint_path: Path
    weight: float
    config: dict
    model_name: str
    checkpoint_epoch: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def parse_path_list(text: str) -> list[Path]:
    values = [Path(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated path.")
    return values


def image_files(image_folder: Path) -> list[Path]:
    return sorted(
        path for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_labels(image_folder: Path) -> dict[str, int] | None:
    labels_path = image_folder / "labels.csv"
    if not labels_path.is_file():
        return None
    with labels_path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: int(row["label"]) for row in csv.DictReader(handle)}


def build_transform(image_size: int) -> transforms.Compose:
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def detections_from_result(result, detection_type):
    if result.boxes is None:
        return []
    names = result.names
    detections = []
    for class_id, confidence, coordinates in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
        strict=True,
    ):
        class_index = int(class_id)
        class_name = names[class_index] if isinstance(names, dict) else names[class_index]
        detections.append(
            detection_type(
                class_name=str(class_name),
                confidence=float(confidence),
                x1=float(coordinates[0]),
                y1=float(coordinates[1]),
                x2=float(coordinates[2]),
                y2=float(coordinates[3]),
            )
        )
    return detections


def make_yolo_cropper(detector_name: str, detector_device: str, confidence: float, padding: float):
    from ultralytics import YOLO

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from animal_recognition.yolo_crop import (
        Detection,
        padded_square_crop_box,
        select_largest_target_animal,
    )

    detector = YOLO(detector_name)

    def crop(path: Path, image: Image.Image) -> Image.Image:
        results = detector.predict(
            source=str(path),
            conf=confidence,
            device=detector_device,
            verbose=False,
        )
        if not results:
            return image
        detection = select_largest_target_animal(detections_from_result(results[0], Detection))
        if detection is None:
            return image
        crop_box = padded_square_crop_box(detection, image.width, image.height, padding)
        return image if crop_box is None else image.crop(crop_box)

    return crop


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from animal_recognition.models import build_model

    config = read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def external_to_internal_labels(labels: dict[str, int]) -> list[int]:
    from animal_recognition.constants import external_to_internal

    return [external_to_internal(label) for label in labels.values()]


def internal_to_external_prediction(label: int) -> int:
    from animal_recognition.constants import internal_to_external

    return internal_to_external(label)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--weights", required=True, help="Comma-separated ensemble weights.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preprocess", choices=["center-crop", "yolo-crop"], default="yolo-crop")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--padding-fraction", type=float, default=0.10)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    config_paths = [resolve_project_path(path) for path in args.config]
    weights = parse_float_list(args.weights)
    if len(weights) != len(config_paths):
        raise ValueError("Number of weights must match number of configs.")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Ensemble weights must sum to 1.0.")

    checkpoint_paths = (
        [resolve_project_path(path) for path in args.checkpoint]
        if args.checkpoint else [
            resolve_project_path(read_json(path)["output_dir"]) / "best.pt"
            for path in config_paths
        ]
    )
    if len(checkpoint_paths) != len(config_paths):
        raise ValueError("Pass either no checkpoints or one checkpoint per config.")

    loaded_models = []
    specs: list[ModelSpec] = []
    for config_path, checkpoint_path, weight in zip(config_paths, checkpoint_paths, weights, strict=True):
        model, checkpoint = load_model(config_path, checkpoint_path, device)
        loaded_models.append(model)
        config = read_json(config_path)
        specs.append(
            ModelSpec(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                weight=weight,
                config=config,
                model_name=str(checkpoint.get("model_name", config["model"]["name"])),
                checkpoint_epoch=int(checkpoint.get("epoch", -1)),
            )
        )

    image_size = int(specs[0].config["data"].get("image_size", 224))
    if any(int(spec.config["data"].get("image_size", 224)) != image_size for spec in specs):
        raise ValueError("All ensemble configs must use the same image size for this script.")
    transform = build_transform(image_size)
    cropper = (
        make_yolo_cropper(
            args.detector,
            args.detector_device,
            args.yolo_confidence,
            args.padding_fraction,
        )
        if args.preprocess == "yolo-crop"
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = image_files(args.image_folder)
    labels = read_labels(args.image_folder)
    predictions: list[int] = []
    targets: list[int] = []
    rows: list[dict[str, object]] = []
    timings: list[float] = []

    with torch.no_grad():
        for image_path in files:
            start = time.perf_counter()
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            if cropper is not None:
                image = cropper(image_path, image)
            tensor = transform(image).unsqueeze(0).to(device)
            ensemble_probabilities = None
            for model, weight in zip(loaded_models, weights, strict=True):
                probabilities = torch.softmax(model(tensor), dim=1)
                weighted = probabilities * weight
                ensemble_probabilities = (
                    weighted if ensemble_probabilities is None else ensemble_probabilities + weighted
                )
            confidence, predicted_internal = torch.max(ensemble_probabilities[0], dim=0)
            predicted_internal_int = int(predicted_internal.item())
            if float(confidence.item()) < args.threshold:
                from animal_recognition.constants import REJECT_INTERNAL

                predicted_internal_int = REJECT_INTERNAL
            predicted_external = internal_to_external_prediction(predicted_internal_int)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            predictions.append(predicted_internal_int)
            row: dict[str, object] = {
                "filename": image_path.name,
                "prediction": predicted_external,
                "confidence": float(confidence.item()),
                "seconds": elapsed,
            }
            if labels is not None and image_path.name in labels:
                from animal_recognition.constants import external_to_internal

                target_external = int(labels[image_path.name])
                targets.append(external_to_internal(target_external))
                row["label"] = target_external
            rows.append(row)

    write_csv(args.output_dir / "predictions.csv", rows)
    summary: dict[str, object] = {
        "device": str(device),
        "image_folder": str(args.image_folder),
        "num_images": len(files),
        "preprocess": args.preprocess,
        "threshold": args.threshold,
        "weights": weights,
        "average_seconds_per_image": statistics.mean(timings) if timings else 0.0,
        "median_seconds_per_image": statistics.median(timings) if timings else 0.0,
        "max_seconds_per_image": max(timings) if timings else 0.0,
        "under_5_seconds_average": (statistics.mean(timings) <= 5.0) if timings else True,
        "models": [
            {
                "config": str(spec.config_path),
                "checkpoint": str(spec.checkpoint_path),
                "weight": spec.weight,
                "model_name": spec.model_name,
                "checkpoint_epoch": spec.checkpoint_epoch,
            }
            for spec in specs
        ],
    }
    if labels is not None and targets:
        from animal_recognition.metrics import (
            classification_metrics,
            per_class_metrics,
            write_confusion_matrix_csv,
            write_confusion_matrix_png,
        )

        metrics = classification_metrics(targets, predictions)
        summary["metrics"] = metrics
        write_confusion_matrix_csv(args.output_dir / "confusion_matrix.csv", targets, predictions)
        write_confusion_matrix_png(
            args.output_dir / "confusion_matrix.png",
            targets,
            predictions,
            f"Official folder ensemble, tau={args.threshold:.2f}",
        )
        write_csv(args.output_dir / "per_class_metrics.csv", per_class_metrics(targets, predictions))

    (args.output_dir / "timing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
