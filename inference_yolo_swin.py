"""YOLO-gated Swin-Tiny inference for the Fine-grained Animal Recognition project.

This keeps the official inference interface but adds a detector gate before the
classifier: if YOLO does not find a cat or dog, the image is rejected; otherwise
the largest detected cat/dog crop is classified by the Swin-Tiny checkpoint.

    python inference_yolo_swin.py --config <config> --checkpoint <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "sep-animal-recognition"
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from animal_recognition.constants import CLASSES, REJECT_EXTERNAL, REJECT_INTERNAL
from animal_recognition.data import evaluation_transform
from animal_recognition.models import build_model

NUM_CLASSES = len(CLASSES)
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "swin_tiny_pretrained_yolo.json"


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_experiment_path(path_text: str) -> Path:
    """Resolve experiment-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else EXPERIMENT_ROOT / path


def padded_box(
    xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    """Expand a detection box by a fraction of its width/height and clamp it."""
    left, top, right, bottom = xyxy
    width = right - left
    height = bottom - top
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction
    image_width, image_height = image_size
    return (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(image_width, int(right + pad_x)),
        min(image_height, int(bottom + pad_y)),
    )


def save_confusion_matrix_plot(
    matrix,
    target_names: list[str],
    output_path: Path,
    colorbar_max: int = 250,
) -> None:
    """Save a high-resolution annotated confusion-matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=colorbar_max)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Count")
    image.set_clim(0, colorbar_max)

    axis.set_xticks(range(len(target_names)))
    axis.set_yticks(range(len(target_names)))
    axis.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(target_names, fontsize=8)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("YOLO + Swin-Tiny Confusion Matrix")

    threshold = colorbar_max / 2
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=6,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=250)
    plt.close(figure)


def print_validation_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> None:
    """Print the validation metrics requested for the YOLO-gated run."""
    target_names = ["reject(-1)"] + list(CLASSES)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    reject_precision, reject_recall, reject_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[REJECT_EXTERNAL], average=None, zero_division=0
    )
    false_accepts = sum(
        true == REJECT_EXTERNAL and predicted != REJECT_EXTERNAL
        for true, predicted in zip(y_true, y_pred, strict=True)
    )
    false_rejects = sum(
        true != REJECT_EXTERNAL and predicted == REJECT_EXTERNAL
        for true, predicted in zip(y_true, y_pred, strict=True)
    )

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {float(macro_f1):.4f}")
    print(f"Weighted-F1: {float(weighted_f1):.4f}")
    print(f"Reject-F1: {float(reject_f1[0]):.4f}")
    print(f"False accepts: {false_accepts}")
    print(f"False rejects: {false_rejects}")
    print(
        "Macro(P/R/F1)="
        f"{float(macro_precision):.4f}/{float(macro_recall):.4f}/{float(macro_f1):.4f}"
    )
    print(
        "Weighted(P/R/F1)="
        f"{float(weighted_precision):.4f}/{float(weighted_recall):.4f}/{float(weighted_f1):.4f}"
    )
    print(
        "Reject(P/R/F1)="
        f"{float(reject_precision[0]):.4f}/{float(reject_recall[0]):.4f}/{float(reject_f1[0]):.4f}"
    )
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


class Model(nn.Module):
    """Reject images without cat/dog detections, then classify the largest crop."""

    def __init__(self, config_path: Path | None = None, checkpoint_path: Path | None = None) -> None:
        super().__init__()
        config_text = os.environ.get("ANIMAL_RECOGNITION_CONFIG")
        self.config_path = Path(config_text) if config_text else (config_path or DEFAULT_CONFIG)
        self.config = read_json(self.config_path)

        output_dir = resolve_experiment_path(str(self.config["output_dir"]))
        checkpoint_text = os.environ.get("ANIMAL_RECOGNITION_CHECKPOINT")
        resolved_checkpoint = (
            Path(checkpoint_text) if checkpoint_text else (checkpoint_path or output_dir / "best.pt")
        )
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                "Swin-Tiny checkpoint was not found. Set ANIMAL_RECOGNITION_CHECKPOINT "
                f"or pass --checkpoint. Looked for: {resolved_checkpoint}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "The YOLO-gated inference requires the 'ultralytics' package."
            ) from error

        yolo_config = self.config["yolo"]
        yolo_model = os.environ.get("ANIMAL_RECOGNITION_YOLO_MODEL", str(yolo_config["model"]))
        self.detector = YOLO(yolo_model)
        self.yolo_confidence = float(yolo_config["confidence_threshold"])
        self.allowed_detection_classes = {str(name) for name in yolo_config["allowed_classes"]}
        self.crop_padding = float(yolo_config["crop_padding"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolved_checkpoint, map_location=self.device)
        checkpoint_config = checkpoint.get("config", self.config)
        model_config = dict(checkpoint_config["model"])
        if bool(model_config.get("pretrained", False)):
            model_config["weights"] = None

        self.classifier = build_model(model_config).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()
        self.transform = evaluation_transform(int(checkpoint_config["data"]["image_size"]))
        self.confidence_threshold = float(
            self.config.get("postprocessing", {}).get("confidence_threshold", 0.0)
        )

    def _largest_cat_or_dog_crop(self, image: Image.Image) -> Image.Image | None:
        results = self.detector.predict(image, conf=self.yolo_confidence, verbose=False)
        if not results:
            return None

        names = results[0].names
        best_box: tuple[float, float, float, float] | None = None
        best_area = 0.0
        for box in results[0].boxes:
            class_index = int(box.cls.item())
            class_name = str(names[class_index])
            if class_name not in self.allowed_detection_classes:
                continue

            left, top, right, bottom = box.xyxy[0].tolist()
            area = max(0.0, right - left) * max(0.0, bottom - top)
            if area > best_area:
                best_area = area
                best_box = (left, top, right, bottom)

        if best_box is None:
            return None
        return image.crop(padded_box(best_box, image.size, self.crop_padding))

    @torch.inference_mode()
    def predict_with_metadata(self, image: Image.Image) -> tuple[int, float, bool]:
        crop = self._largest_cat_or_dog_crop(image.convert("RGB"))
        if crop is None:
            return REJECT_EXTERNAL, 0.0, False

        image_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.classifier(image_tensor), dim=1)
        confidence, internal_label = probabilities.max(dim=1)
        confidence_value = float(confidence.item())
        if float(confidence.item()) < self.confidence_threshold:
            return REJECT_EXTERNAL, confidence_value, True

        label = int(internal_label.item())
        external_label = REJECT_EXTERNAL if label == REJECT_INTERNAL else label
        return external_label, confidence_value, True

    @torch.inference_mode()
    def forward(self, image: Image.Image) -> int:
        label, _, _ = self.predict_with_metadata(image)
        return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("splits/val_seed42.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/all"))
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("runs/swin_tiny_pretrained_yolo/yolo_swin_predictions.csv"),
    )
    parser.add_argument(
        "--confusion-csv",
        type=Path,
        default=Path("runs/swin_tiny_pretrained_yolo/yolo_swin_confusion_matrix.csv"),
    )
    parser.add_argument(
        "--confusion-plot",
        type=Path,
        default=Path("runs/swin_tiny_pretrained_yolo/yolo_swin_confusion_matrix.png"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    dataset_root = args.dataset_root
    df = pd.read_csv(manifest_path)
    model = Model(config_path=args.config, checkpoint_path=args.checkpoint).eval()

    prediction_rows = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image_path = dataset_root / filename
            if not image_path.is_file():
                print(f"Warning: missing image, skipping: {image_path}")
                continue

            try:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")
            except OSError as error:
                print(f"Warning: could not open image, skipping: {image_path} ({error})")
                continue

            pred, confidence, yolo_detected = model.predict_with_metadata(image)
            true_label = int(label)
            predicted_label = int(pred)
            y_true.append(true_label)
            y_pred.append(predicted_label)
            prediction_rows.append(
                {
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "yolo_detected": yolo_detected,
                }
            )

    if not prediction_rows:
        raise RuntimeError("No validation images were evaluated. Check --manifest and --dataset-root.")

    labels = [REJECT_EXTERNAL] + list(range(20))
    target_names = ["reject(-1)"] + list(CLASSES)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(args.predictions_csv, index=False)

    args.confusion_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    matrix_df.index.name = "true_label"
    matrix_df.to_csv(args.confusion_csv)

    save_confusion_matrix_plot(matrix, target_names, args.confusion_plot, colorbar_max=250)
    print_validation_metrics(y_true, y_pred, labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(matrix)
    print(f"Saved predictions CSV to: {args.predictions_csv.resolve()}")
    print(f"Saved confusion matrix CSV to: {args.confusion_csv.resolve()}")
    print(f"Saved confusion matrix plot to: {args.confusion_plot.resolve()}")
