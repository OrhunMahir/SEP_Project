#!/usr/bin/env python3
"""Cache largest YOLO cat/dog crops for training and validation manifests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from animal_recognition.data import Sample, load_split
from animal_recognition.yolo_crop import (
    Detection,
    clamp_crop_box,
    padded_square_crop_box,
    select_largest_target_animal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def batched(values: list[Sample], batch_size: int):
    """Yield contiguous batches without an extra runtime dependency."""
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


def destination_for(sample: Sample, output_root: Path) -> Path:
    """Map one manifest-relative source path to its cached output path."""
    return output_root / sample.relative_path


def save_crop_or_fallback(
    sample: Sample,
    detection: Detection | None,
    output_path: Path,
    padding_fraction: float,
    square_crop: bool,
) -> bool:
    """Save a largest-animal crop or copy the source image when no crop is valid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if detection is None:
        shutil.copy2(sample.path, output_path)
        return False

    with Image.open(sample.path) as opened_image:
        image = opened_image.convert("RGB")
        crop_box = (
            padded_square_crop_box(
                detection,
                image.width,
                image.height,
                padding_fraction,
            )
            if square_crop
            else clamp_crop_box(detection, image.width, image.height)
        )
        if crop_box is None:
            shutil.copy2(sample.path, output_path)
            return False
        cropped = image.crop(crop_box)
        if output_path.suffix.lower() == ".png":
            cropped.save(output_path, format="PNG")
        else:
            cropped.save(output_path, format="JPEG", quality=95)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "efficientnet_b0_yolo_crop.json",
    )
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--padding-fraction", type=float, default=0.0)
    parser.add_argument("--square-crop", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in the interval [0.0, 1.0].")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.padding_fraction < 0.0:
        raise ValueError("padding-fraction must be non-negative.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for YOLO crop preparation. Install requirements-yolo.txt first."
        ) from error

    config = read_json(args.config)
    data_config = config["data"]
    if "image_root" not in data_config:
        raise ValueError("The crop-training config must define data.image_root.")
    output_root = resolve_project_path(str(data_config["image_root"]))
    runs_root = (PROJECT_ROOT / "runs").resolve()
    try:
        output_root.resolve().relative_to(runs_root)
    except ValueError as error:
        raise ValueError("YOLO crop output must stay inside the project's runs directory.") from error

    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    source_root = Path(data_paths["train_image_root"])
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source image root was not found: {source_root}")
    if output_root.resolve() == source_root.resolve():
        raise ValueError("YOLO crop output must not overwrite the source dataset.")

    train_samples = load_split(resolve_project_path(data_config["train_split"]), source_root)
    validation_samples = load_split(resolve_project_path(data_config["validation_split"]), source_root)
    all_samples = train_samples + validation_samples
    pending_samples = [
        sample
        for sample in all_samples
        if args.overwrite or not destination_for(sample, output_root).is_file()
    ]

    detector = YOLO(args.detector)
    crop_count = 0
    fallback_count = 0
    for sample_batch in batched(pending_samples, args.batch_size):
        results = detector.predict(
            source=[str(sample.path) for sample in sample_batch],
            conf=args.confidence,
            device=args.detector_device,
            verbose=False,
        )
        if len(results) != len(sample_batch):
            raise RuntimeError("YOLO returned an unexpected number of detection results.")
        for sample, result in zip(sample_batch, results):
            detection = select_largest_target_animal(detections_from_result(result))
            used_crop = save_crop_or_fallback(
                sample,
                detection,
                destination_for(sample, output_root),
                args.padding_fraction,
                args.square_crop,
            )
            crop_count += int(used_crop)
            fallback_count += int(not used_crop)

    summary = {
        "experiment_name": config["experiment_name"],
        "detector": args.detector,
        "detector_confidence": args.confidence,
        "padding_fraction": args.padding_fraction,
        "square_crop": args.square_crop,
        "source_image_root": str(source_root),
        "output_image_root": str(output_root),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "processed_samples": len(pending_samples),
        "skipped_existing_samples": len(all_samples) - len(pending_samples),
        "largest_cat_or_dog_crops": crop_count,
        "fallback_raw_images": fallback_count,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    (output_root.parent / "yolo_crop_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
