#!/usr/bin/env python3
"""Generate one Grad-CAM visualization per animal breed for Custom CNN checkpoints.

This script is intentionally standalone: it imports the trained project model, but
does not modify project training, inference, or dataset files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Candidate:
    path: Path
    class_index: int
    class_name: str


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    confidence: float
    predicted_index: int
    predicted_name: str
    logits: torch.Tensor


class MissingCropError(RuntimeError):
    """Raised when YOLO preprocessing cannot produce a valid animal crop."""


class GradCAM:
    """Minimal Grad-CAM implementation for CNN feature maps."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, inputs: torch.Tensor, target_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        score = logits[:, target_index].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap[0, 0]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if float(heatmap_max - heatmap_min) > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = torch.zeros_like(heatmap)
        return heatmap.detach().cpu(), logits.detach().cpu()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


def image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def read_manifest_candidates(
    dataset_root: Path,
    labels_csv: Path,
    classes: tuple[str, ...],
    include_reject: bool,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels CSV must contain filename and label columns.")
        for row in reader:
            label = int(row["label"])
            if label < 0:
                if not include_reject:
                    continue
                class_index = len(classes)
                class_name = "reject"
            else:
                if label >= len(classes):
                    raise ValueError(f"Unsupported class label in CSV: {label}")
                class_index = label
                class_name = classes[label]
            candidates.append(Candidate(dataset_root / row["filename"], class_index, class_name))
    return candidates


def normalized_name(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip()


def read_folder_candidates(
    dataset_root: Path,
    classes: tuple[str, ...],
    include_reject: bool,
) -> list[Candidate]:
    name_to_index = {normalized_name(name): index for index, name in enumerate(classes)}
    if include_reject:
        name_to_index["reject"] = len(classes)
    candidates: list[Candidate] = []
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        class_index = name_to_index.get(normalized_name(folder.name))
        if class_index is None:
            continue
        class_name = "reject" if class_index == len(classes) else classes[class_index]
        for image_path in image_files(folder):
            candidates.append(Candidate(image_path, class_index, class_name))
    return candidates


def load_candidates(
    dataset_root: Path,
    labels_csv: Path | None,
    classes: tuple[str, ...],
    include_reject: bool,
) -> list[Candidate]:
    if labels_csv is not None:
        return read_manifest_candidates(dataset_root, labels_csv, classes, include_reject)
    return read_folder_candidates(dataset_root, classes, include_reject)


def build_transform(image_size: int) -> transforms.Compose:
    from torchvision import transforms

    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def detections_from_result(result, detection_type):
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


def make_yolo_cropper(args, project_root: Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for --preprocess yolo-crop. "
            "Run this on the same environment used for YOLO crop training."
        ) from error

    sys.path.insert(0, str(project_root / "src"))
    from animal_recognition.yolo_crop import (
        Detection,
        padded_square_crop_box,
        select_largest_target_animal,
    )

    detector = YOLO(args.detector)

    def crop(path: Path, image: Image.Image) -> Image.Image | None:
        results = detector.predict(
            source=str(path),
            conf=args.yolo_confidence,
            device=args.detector_device,
            verbose=False,
        )
        if not results:
            return None
        detection = select_largest_target_animal(detections_from_result(results[0], Detection))
        if detection is None:
            return None
        crop_box = padded_square_crop_box(
            detection,
            image.width,
            image.height,
            args.padding_fraction,
        )
        if crop_box is None:
            return None
        return image.crop(crop_box)

    return crop


def preprocess_image(
    path: Path,
    transform: transforms.Compose,
    cropper=None,
    on_yolo_miss: str = "skip",
) -> tuple[Image.Image, torch.Tensor]:
    from torchvision import transforms

    with Image.open(path) as opened:
        original = opened.convert("RGB")
    image = original
    if cropper is not None:
        cropped = cropper(path, original)
        if cropped is None:
            if on_yolo_miss == "skip":
                raise MissingCropError(f"No valid YOLO crop for {path}")
            image = original
        else:
            image = cropped
    tensor = transform(image)
    preview_transform = transforms.Compose(transform.transforms[:2])
    preview = preview_transform(image)
    return preview, tensor.unsqueeze(0)


def get_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    module: torch.nn.Module = model
    for part in module_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def colorize_heatmap(heatmap: torch.Tensor) -> Image.Image:
    heatmap_image = Image.fromarray((heatmap.numpy() * 255).astype("uint8"), mode="L")
    width, height = heatmap_image.size
    color = Image.new("RGBA", (width, height))
    pixels = color.load()
    gray = heatmap_image.load()
    for y in range(height):
        for x in range(width):
            value = gray[x, y] / 255.0
            r = int(255 * min(1.0, value * 1.7))
            g = int(255 * max(0.0, 1.0 - abs(value - 0.50) * 2.0))
            b = int(255 * max(0.0, 1.0 - value * 1.6))
            pixels[x, y] = (r, g, b, int(190 * value))
    return color


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.append("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_overlay(
    original: Image.Image,
    heatmap: torch.Tensor,
    candidate: ScoredCandidate,
    target_name: str,
    output_path: Path,
) -> None:
    base = original.resize((384, 384))
    heat = colorize_heatmap(heatmap).resize(base.size)
    overlay = Image.alpha_composite(base.convert("RGBA"), heat)
    canvas = Image.new("RGB", (384, 472), "white")
    canvas.paste(overlay.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = font(18, True)
    small_font = font(14)
    draw.text((14, 398), target_name.replace("_", " "), fill=(31, 41, 55), font=title_font)
    subtitle = (
        f"pred={candidate.predicted_name.replace('_', ' ')} | "
        f"target confidence={candidate.confidence:.3f}"
    )
    draw.text((14, 426), subtitle, fill=(75, 85, 99), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def score_candidates(
    model: torch.nn.Module,
    candidates: list[Candidate],
    classes: tuple[str, ...],
    transform: transforms.Compose,
    device: torch.device,
    max_candidates_per_class: int,
    cropper=None,
    on_yolo_miss: str = "skip",
) -> dict[int, ScoredCandidate]:
    import torch

    by_class: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_class.setdefault(candidate.class_index, []).append(candidate)

    selected: dict[int, ScoredCandidate] = {}
    model.eval()
    with torch.no_grad():
        for class_index, class_candidates in sorted(by_class.items()):
            best_correct: ScoredCandidate | None = None
            best_any: ScoredCandidate | None = None
            for candidate in class_candidates[:max_candidates_per_class]:
                if not candidate.path.is_file():
                    continue
                try:
                    _preview, tensor = preprocess_image(
                        candidate.path,
                        transform,
                        cropper=cropper,
                        on_yolo_miss=on_yolo_miss,
                    )
                except MissingCropError:
                    continue
                logits = model(tensor.to(device)).cpu()[0]
                probabilities = torch.softmax(logits, dim=0)
                confidence = float(probabilities[class_index])
                predicted_index = int(probabilities.argmax())
                predicted_name = "reject" if predicted_index == len(classes) else classes[predicted_index]
                scored = ScoredCandidate(candidate, confidence, predicted_index, predicted_name, logits)
                if best_any is None or scored.confidence > best_any.confidence:
                    best_any = scored
                if predicted_index == class_index and (
                    best_correct is None or scored.confidence > best_correct.confidence
                ):
                    best_correct = scored
            if best_correct is not None:
                selected[class_index] = best_correct
            elif best_any is not None:
                selected[class_index] = best_any
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-layer", default="features.3.layers.3")
    parser.add_argument("--max-candidates-per-class", type=int, default=200)
    parser.add_argument("--include-reject", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preprocess", choices=["center-crop", "yolo-crop"], default="center-crop")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--padding-fraction", type=float, default=0.10)
    parser.add_argument("--on-yolo-miss", choices=["skip", "center-crop"], default="skip")
    args = parser.parse_args()

    import torch

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))

    from animal_recognition.constants import CLASSES
    from animal_recognition.models import build_model

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = read_json(args.config) if args.config is not None else checkpoint["config"]
    checkpoint_config = checkpoint.get("config", config)

    model = build_model(checkpoint_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(checkpoint_config["data"].get("image_size", 224))
    transform = build_transform(image_size)
    cropper = make_yolo_cropper(args, project_root) if args.preprocess == "yolo-crop" else None
    classes = tuple(CLASSES)
    candidates = load_candidates(args.dataset_root, args.labels_csv, classes, args.include_reject)
    selected = score_candidates(
        model,
        candidates,
        classes,
        transform,
        device,
        args.max_candidates_per_class,
        cropper=cropper,
        on_yolo_miss=args.on_yolo_miss,
    )
    target_layer = get_module(model, args.target_layer)
    gradcam = GradCAM(model, target_layer)

    summary: list[dict[str, object]] = []
    expected_indices = range(len(classes) + int(args.include_reject))
    try:
        for class_index in expected_indices:
            class_name = "reject" if class_index == len(classes) else classes[class_index]
            scored = selected.get(class_index)
            if scored is None:
                summary.append({"class_name": class_name, "status": "missing_candidate"})
                continue
            preview, tensor = preprocess_image(
                scored.candidate.path,
                transform,
                cropper=cropper,
                on_yolo_miss=args.on_yolo_miss,
            )
            heatmap, logits = gradcam(tensor.to(device), class_index)
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_index = int(probabilities.argmax())
            predicted_name = "reject" if predicted_index == len(classes) else classes[predicted_index]
            updated = ScoredCandidate(
                scored.candidate,
                float(probabilities[class_index]),
                predicted_index,
                predicted_name,
                logits[0],
            )
            output_path = args.output_dir / f"{class_index:02d}_{slugify(class_name)}_gradcam.png"
            render_overlay(preview, heatmap, updated, class_name, output_path)
            summary.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "source_image": str(scored.candidate.path),
                    "output_image": str(output_path),
                    "target_confidence": updated.confidence,
                    "predicted_index": predicted_index,
                    "predicted_name": predicted_name,
                    "correct_prediction": predicted_index == class_index,
                    "preprocess": args.preprocess,
                }
            )
    finally:
        gradcam.remove()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gradcam_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {sum('output_image' in row for row in summary)} Grad-CAM images to: {args.output_dir}")
    missing = [row["class_name"] for row in summary if row.get("status") == "missing_candidate"]
    if missing:
        print("Missing classes:", ", ".join(str(item) for item in missing))


if __name__ == "__main__":
    main()
