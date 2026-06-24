#!/usr/bin/env python3
"""Compare raw and YOLO-cropped validation inference for a fixed classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from animal_recognition.data import evaluation_transform, load_split
from animal_recognition.metrics import classification_metrics
from animal_recognition.models import build_model
from animal_recognition.thresholding import apply_confidence_threshold
from animal_recognition.yolo_crop import Detection, clamp_crop_box, select_largest_target_animal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def batched(values: list, batch_size: int):
    """Yield contiguous batches without requiring an additional dependency."""
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def detections_from_result(result) -> list[Detection]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "resnet18_scratch.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "resnet18_yolo_crop")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in the interval [0.0, 1.0].")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in the interval [0.0, 1.0].")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for the YOLO ablation. Install requirements-yolo.txt first."
        ) from error

    config = read_json(args.config)
    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    classifier_device = torch.device(args.device)
    checkpoint_path = args.checkpoint or resolve_project_path(config["output_dir"]) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=classifier_device)
    checkpoint_config = checkpoint["config"]
    if checkpoint_config["model"]["name"] != "resnet18":
        raise ValueError("This ablation is restricted to the selected ResNet-18 checkpoint.")
    if float(checkpoint_config["model"].get("dropout", 0.0)) != 0.0:
        raise ValueError("This ablation requires the selected ResNet-18 dropout=0.0 checkpoint.")

    classifier = build_model(checkpoint_config["model"]).to(classifier_device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    data_config = checkpoint_config["data"]
    validation_samples = load_split(
        resolve_project_path(data_config["validation_split"]),
        Path(data_paths["train_image_root"]),
    )
    transform = evaluation_transform(int(data_config["image_size"]))
    detector = YOLO(args.detector)

    raw_probability_batches: list[torch.Tensor] = []
    crop_probability_batches: list[torch.Tensor] = []
    targets: list[int] = []
    detected_animal_count = 0

    for sample_batch in batched(validation_samples, args.batch_size):
        detector_results = detector.predict(
            source=[str(sample.path) for sample in sample_batch],
            conf=args.confidence,
            device=args.detector_device,
            verbose=False,
        )
        raw_images: list[torch.Tensor] = []
        cropped_images: list[torch.Tensor] = []
        for sample, result in zip(sample_batch, detector_results):
            with Image.open(sample.path) as opened_image:
                image = opened_image.convert("RGB")
                detection = select_largest_target_animal(detections_from_result(result))
                crop_box = (
                    clamp_crop_box(detection, image.width, image.height) if detection is not None else None
                )
                if crop_box is None:
                    cropped_image = image
                else:
                    cropped_image = image.crop(crop_box)
                    detected_animal_count += 1
                raw_images.append(transform(image))
                cropped_images.append(transform(cropped_image))
                targets.append(sample.label)

        with torch.no_grad():
            raw_logits = classifier(torch.stack(raw_images).to(classifier_device, non_blocking=True))
            crop_logits = classifier(torch.stack(cropped_images).to(classifier_device, non_blocking=True))
        raw_probability_batches.append(torch.softmax(raw_logits, dim=1).cpu())
        crop_probability_batches.append(torch.softmax(crop_logits, dim=1).cpu())

    raw_predictions, _ = apply_confidence_threshold(
        torch.cat(raw_probability_batches), args.threshold
    )
    crop_predictions, _ = apply_confidence_threshold(
        torch.cat(crop_probability_batches), args.threshold
    )
    raw_metrics = classification_metrics(targets, raw_predictions.tolist())
    crop_metrics = classification_metrics(targets, crop_predictions.tolist())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_name": "resnet18_yolo_crop_ablation",
        "classifier_checkpoint": str(checkpoint_path),
        "classifier_checkpoint_epoch": int(checkpoint["epoch"]),
        "classifier_model": checkpoint_config["model"],
        "threshold": args.threshold,
        "detector": args.detector,
        "detector_confidence": args.confidence,
        "validation_samples": len(targets),
        "largest_cat_or_dog_crops": detected_animal_count,
        "raw_image_metrics": raw_metrics,
        "yolo_crop_metrics": crop_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Classifier checkpoint epoch: {checkpoint['epoch']}")
    print(f"Fixed threshold: {args.threshold:.2f}")
    print(f"Largest cat/dog crops: {detected_animal_count}/{len(targets)}")
    print(
        "Raw image | "
        f"accuracy={float(raw_metrics['accuracy']):.4f} | "
        f"macro_f1={float(raw_metrics['macro_f1']):.4f} | "
        f"reject_f1={float(raw_metrics['reject_f1']):.4f} | "
        f"false_accepts={int(raw_metrics['false_accepts'])} | "
        f"false_rejects={int(raw_metrics['false_rejects'])}"
    )
    print(
        "YOLO crop | "
        f"accuracy={float(crop_metrics['accuracy']):.4f} | "
        f"macro_f1={float(crop_metrics['macro_f1']):.4f} | "
        f"reject_f1={float(crop_metrics['reject_f1']):.4f} | "
        f"false_accepts={int(crop_metrics['false_accepts'])} | "
        f"false_rejects={int(crop_metrics['false_rejects'])}"
    )


if __name__ == "__main__":
    main()
