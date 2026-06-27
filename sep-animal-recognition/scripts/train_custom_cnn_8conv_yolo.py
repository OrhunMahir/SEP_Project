#!/usr/bin/env python3
"""Train the 8-conv CNN on YOLO-cropped cat/dog regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from animal_recognition.constants import NUM_OUTPUTS
from animal_recognition.data import (
    BalancedEpochSampler,
    Sample,
    evaluation_transform,
    load_split,
)
from animal_recognition.models import count_trainable_parameters
from animal_recognition.training import (
    evaluate_model,
    set_seed,
    set_warmup_cosine_learning_rate,
    train_one_epoch,
)
from train_custom_cnn_8conv import CustomCNN8Conv
from train_custom_cnn_8conv_strong_aug import strong_training_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def padded_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> list[int]:
    """Return a padded box clipped to image bounds."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(image_width, int(round(x2 + pad_x))),
        min(image_height, int(round(y2 + pad_y))),
    ]


def yolo_cache_path(config: dict, output_dir: Path) -> Path:
    """Resolve the YOLO box-cache location."""
    configured_path = config.get("yolo", {}).get("box_cache")
    if configured_path:
        return resolve_project_path(str(configured_path))
    return output_dir / "yolo_boxes.json"


def load_or_create_yolo_boxes(
    samples: Sequence[Sample],
    config: dict,
    output_dir: Path,
    device: torch.device,
    force: bool = False,
) -> dict[str, dict[str, object]]:
    """Load cached YOLO crop boxes or create them for the provided samples."""
    cache_path = yolo_cache_path(config, output_dir)
    if cache_path.is_file() and not force:
        return read_json(cache_path)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "ultralytics is required for YOLO cropping. Install it in the cluster "
            "environment or provide a precomputed yolo_boxes.json cache."
        ) from error

    yolo_config = config["yolo"]
    model = YOLO(str(yolo_config["model"]))
    confidence = float(yolo_config.get("confidence", 0.25))
    padding = float(yolo_config.get("padding", 0.10))
    class_ids = [int(class_id) for class_id in yolo_config.get("class_ids", [15, 16])]
    yolo_device = 0 if device.type == "cuda" else "cpu"

    records: dict[str, dict[str, object]] = {}
    unique_samples = {sample.relative_path: sample for sample in samples}
    print(
        "Preparing YOLO crop cache: "
        f"{len(unique_samples)} unique images | conf={confidence} | padding={padding}"
    )
    for index, sample in enumerate(unique_samples.values(), start=1):
        with Image.open(sample.path) as image:
            image_width, image_height = image.size

        results = model.predict(
            source=str(sample.path),
            conf=confidence,
            classes=class_ids,
            device=yolo_device,
            verbose=False,
        )
        best_record: dict[str, object] = {
            "box": None,
            "confidence": None,
            "class_id": None,
            "source": "full_image",
        }
        best_area = -1.0
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for detected_box in boxes:
                    xyxy = [float(value) for value in detected_box.xyxy[0].tolist()]
                    area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                    if area > best_area:
                        best_area = area
                        best_record = {
                            "box": padded_box(xyxy, image_width, image_height, padding),
                            "confidence": float(detected_box.conf[0].item()),
                            "class_id": int(detected_box.cls[0].item()),
                            "source": "yolo",
                        }
        records[sample.relative_path] = best_record
        if index % 250 == 0 or index == len(unique_samples):
            print(f"YOLO cache progress: {index}/{len(unique_samples)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    detections = sum(record["box"] is not None for record in records.values())
    print(f"Saved YOLO crop cache to: {cache_path}")
    print(f"YOLO detections used: {detections}/{len(records)}")
    return records


class YoloCropDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Dataset that crops each image with a precomputed YOLO cat/dog box."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform,
        crop_records: dict[str, dict[str, object]],
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.crop_records = crop_records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            crop_record = self.crop_records.get(sample.relative_path, {})
            box = crop_record.get("box")
            if box is not None:
                image = image.crop(tuple(int(value) for value in box))
            tensor = self.transform(image)
        return tensor, sample.label, sample.relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "custom_cnn_8conv_yolo_medium_aug.json",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--refresh-yolo-cache", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    model_config = config["model"]
    if model_config["name"] != "custom_cnn_8conv_yolo":
        raise ValueError("This script only supports model.name='custom_cnn_8conv_yolo'.")

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
    augmentation_config = config["augmentation"]
    max_epochs = args.max_epochs or int(training_config["max_epochs"])
    selection_metric = str(training_config.get("selection_metric", "macro_f1"))
    if selection_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("selection_metric must be either 'accuracy' or 'macro_f1'.")

    output_dir = resolve_project_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    image_root = resolve_project_path(str(data_paths["train_image_root"]))
    train_samples = load_split(
        resolve_project_path(str(data_config["train_split"])), image_root
    )
    validation_samples = load_split(
        resolve_project_path(str(data_config["validation_split"])), image_root
    )
    crop_records = load_or_create_yolo_boxes(
        samples=[*train_samples, *validation_samples],
        config=config,
        output_dir=output_dir,
        device=device,
        force=args.refresh_yolo_cache,
    )

    image_size = int(data_config["image_size"])
    train_dataset = YoloCropDataset(
        train_samples,
        strong_training_transform(image_size, augmentation_config),
        crop_records,
    )
    validation_dataset = YoloCropDataset(
        validation_samples,
        evaluation_transform(image_size),
        crop_records,
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

    model = CustomCNN8Conv(
        num_outputs=int(model_config["num_outputs"]),
        dropout=float(model_config.get("dropout", 0.3)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    best_selection_score = float("-inf")
    best_validation_metrics: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    print(f"Device: {device}")
    print("Preprocessing: YOLO crop largest cat/dog detection")
    print("Architecture: 8 convolution layers (2 per feature block, 3x3 kernels)")
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
                    "model_name": "custom_cnn_8conv_yolo",
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

    yolo_records = read_json(yolo_cache_path(config, output_dir))
    summary = {
        "experiment_name": config["experiment_name"],
        "architecture": "custom_cnn_8conv_yolo",
        "preprocessing": {
            "type": "yolo_largest_cat_dog_crop",
            "confidence": float(config["yolo"].get("confidence", 0.25)),
            "padding": float(config["yolo"].get("padding", 0.10)),
            "detections_used": sum(record["box"] is not None for record in yolo_records.values()),
            "images_cached": len(yolo_records),
        },
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
