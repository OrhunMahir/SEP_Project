#!/usr/bin/env python3
"""Course-style inference entry point for the final animal-recognition ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_recognition.constants import (  # noqa: E402
    CLASSES,
    REJECT_EXTERNAL,
    REJECT_INTERNAL,
    internal_to_external,
)
from animal_recognition.models import build_model  # noqa: E402
from animal_recognition.yolo_crop import (  # noqa: E402
    Detection,
    padded_square_crop_box,
    select_largest_target_animal,
)

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REJECT = REJECT_EXTERNAL
NUM_CLASSES = len(CLASSES)

ENSEMBLE_PRESETS = {
    "pretrained_50ep": {
        "configs": [
            "configs/resnet18_pretrained_yolo_crop_padded_50ep.json",
            "configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json",
            "configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json",
        ],
        "weights": [0.35, 0.35, 0.30],
        "threshold": 0.30,
    },
    "scratch_100ep": {
        "configs": [
            "configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json",
            "configs/resnet18_yolo_crop_padded_100ep.json",
            "configs/efficientnet_b0_yolo_crop_padded_100ep.json",
        ],
        "weights": [0.45, 0.25, 0.30],
        "threshold": 0.32,
    },
}


@dataclass(frozen=True)
class LabelRow:
    filename: str
    label: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_transform(image_size: int) -> transforms.Compose:
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def detections_from_result(result) -> list[Detection]:
    if result.boxes is None:
        return []
    names = result.names
    detections = []
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


def read_label_rows(labels_path: Path) -> list[LabelRow]:
    with labels_path.open(newline="", encoding="utf-8") as handle:
        return [LabelRow(row["filename"], int(row["label"])) for row in csv.DictReader(handle)]


def image_files(image_folder: Path) -> list[Path]:
    return sorted(
        path for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


class Model(nn.Module):
    """Final weighted ensemble with YOLO padded-crop preprocessing."""

    def __init__(
        self,
        preset: str | None = None,
        device: str | torch.device | None = None,
        preprocess: str = "yolo-crop",
        detector: str = "yolov8n.pt",
        detector_device: str | None = None,
        yolo_confidence: float = 0.25,
        padding_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        selected_preset = preset or os.environ.get("ANIMAL_RECOGNITION_PRESET", "pretrained_50ep")
        if selected_preset not in ENSEMBLE_PRESETS:
            raise ValueError(f"Unsupported preset: {selected_preset}")
        if preprocess not in {"center-crop", "yolo-crop"}:
            raise ValueError("preprocess must be 'center-crop' or 'yolo-crop'.")

        if device is None:
            device_name = os.environ.get("ANIMAL_RECOGNITION_DEVICE")
            if device_name is None:
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        preset_info = ENSEMBLE_PRESETS[selected_preset]
        self.preset = selected_preset
        self.weights = [float(weight) for weight in preset_info["weights"]]
        self.threshold = float(preset_info["threshold"])
        self.preprocess = preprocess
        self.detector_device = detector_device or os.environ.get(
            "ANIMAL_RECOGNITION_DETECTOR_DEVICE",
            "0" if self.device.type == "cuda" else "cpu",
        )
        self.yolo_confidence = yolo_confidence
        self.padding_fraction = padding_fraction
        self.models = nn.ModuleList()

        image_size: int | None = None
        for config_name in preset_info["configs"]:
            config_path = resolve_project_path(config_name)
            config = read_json(config_path)
            checkpoint_path = resolve_project_path(config["output_dir"]) / "best.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint for {selected_preset}: {checkpoint_path}"
                )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_config = checkpoint.get("config", config)
            model = build_model(checkpoint_config["model"]).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)

            data_config = checkpoint_config.get("data", config["data"])
            current_image_size = int(data_config.get("image_size", 224))
            if image_size is None:
                image_size = current_image_size
            elif current_image_size != image_size:
                raise ValueError("All ensemble models must use the same image size.")

        self.transform = build_transform(image_size or 224)
        self.detector_model = None
        if self.preprocess == "yolo-crop":
            from ultralytics import YOLO

            self.detector_model = YOLO(detector)

    def crop_image(self, image: Image.Image) -> Image.Image:
        if self.detector_model is None:
            return image
        results = self.detector_model.predict(
            source=image,
            conf=self.yolo_confidence,
            device=self.detector_device,
            verbose=False,
        )
        if not results:
            return image
        detection = select_largest_target_animal(detections_from_result(results[0]))
        if detection is None:
            return image
        crop_box = padded_square_crop_box(
            detection,
            image.width,
            image.height,
            self.padding_fraction,
        )
        return image if crop_box is None else image.crop(crop_box)

    def forward(self, image: Image.Image) -> int:
        image = image.convert("RGB")
        image = self.crop_image(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        ensemble_probabilities = None
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights, strict=True):
                probabilities = torch.softmax(model(tensor), dim=1)
                weighted = probabilities * weight
                ensemble_probabilities = (
                    weighted if ensemble_probabilities is None else ensemble_probabilities + weighted
                )
        confidence, predicted_internal = torch.max(ensemble_probabilities[0], dim=0)
        predicted_label = int(predicted_internal.item())
        if float(confidence.item()) < self.threshold:
            predicted_label = REJECT_INTERNAL
        return int(internal_to_external(predicted_label))


def run_folder(
    image_folder: Path,
    model: Model,
    labels_csv: Path | None = None,
) -> tuple[list[int], list[int] | None]:
    labels_path = labels_csv or image_folder / "labels.csv"
    if labels_path.is_file():
        rows = read_label_rows(labels_path)
        files = [image_folder / row.filename for row in rows]
        y_true: list[int] | None = [row.label for row in rows]
    else:
        files = image_files(image_folder)
        y_true = None

    y_pred = []
    for image_path in files:
        with Image.open(image_path) as image:
            y_pred.append(model(image.convert("RGB")))
        print(f"{image_path.name},{y_pred[-1]}")
    return y_pred, y_true


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default=Path("images"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(ENSEMBLE_PRESETS), default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--preprocess", choices=["center-crop", "yolo-crop"], default="yolo-crop")
    parser.add_argument("--detector-device", default=None)
    args = parser.parse_args()

    model = Model(
        preset=args.preset,
        device=args.device,
        preprocess=args.preprocess,
        detector_device=args.detector_device,
    ).eval()
    y_pred, y_true = run_folder(args.image_folder, model, args.labels_csv)
    if y_true is None:
        return

    labels = [REJECT_EXTERNAL, *range(len(CLASSES))]
    target_names = ["reject(-1)", *CLASSES]
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=3,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    main()
