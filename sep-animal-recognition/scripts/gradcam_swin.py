#!/usr/bin/env python3
"""Create Swin-Tiny Grad-CAM visualizations with optional YOLO preprocessing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import swin_t

from animal_recognition.constants import (
    CLASSES,
    NUM_OUTPUTS,
    REJECT_EXTERNAL,
    REJECT_INTERNAL,
    external_to_internal,
    internal_to_external,
)
from animal_recognition.data import NORMALIZE_MEAN, NORMALIZE_STD
from animal_recognition.gradcam import SwinGradCAM, resolve_swin_target_layer
from animal_recognition.models import SwinTiny
from animal_recognition.yolo_crop import (
    Detection,
    clamp_crop_box,
    padded_square_crop_box,
    select_largest_target_animal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InputItem:
    """One image selected directly or through a labelled manifest."""

    path: Path
    relative_name: str
    true_external_label: int | None


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path: Path | str) -> Path:
    """Resolve project-relative paths without modifying absolute paths."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def class_name(internal_label: int) -> str:
    """Return the report name for an internal classifier output."""
    return "reject" if internal_label == REJECT_INTERNAL else CLASSES[internal_label]


def parse_target_class(
    specification: str,
    true_external_label: int | None,
) -> int | None:
    """Resolve predicted, true, numeric, reject, or class-name targets."""
    normalized = specification.strip()

    if normalized.casefold() == "predicted":
        return None

    if normalized.casefold() == "true":
        if true_external_label is None:
            raise ValueError(
                "--target-class true requires labels supplied by --manifest."
            )
        return external_to_internal(true_external_label)

    if normalized.casefold() in {"reject", "reject(-1)"}:
        return REJECT_INTERNAL

    try:
        numeric_label = int(normalized)
    except ValueError:
        normalized_name = normalized.replace(" ", "_").casefold()
        matches = [
            index
            for index, name in enumerate(CLASSES)
            if name.casefold() == normalized_name
        ]

        if not matches:
            raise ValueError(f"Unknown target class: {specification}")

        return matches[0]

    if numeric_label == REJECT_INTERNAL:
        return REJECT_INTERNAL

    return external_to_internal(numeric_label)


def load_swin_checkpoint(
    checkpoint_path: Path,
    fallback_config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Build the checkpoint architecture without downloading weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain model_state_dict: {checkpoint_path}"
        )

    checkpoint_config = checkpoint.get("config")
    effective_config = (
        checkpoint_config
        if isinstance(checkpoint_config, dict)
        else fallback_config
    )

    model_config = effective_config.get("model", {})

    if model_config.get("name") != "swin_tiny":
        raise ValueError(
            "Expected a swin_tiny checkpoint, received: "
            f"{model_config.get('name')}"
        )

    num_outputs = int(model_config.get("num_outputs", NUM_OUTPUTS))
    dropout = float(model_config.get("dropout", 0.1))
    attention_dropout = float(
        model_config.get("attention_dropout", 0.0)
    )

    state_dict = dict(checkpoint["model_state_dict"])

    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    wrapped_checkpoint = any(
        key.startswith("network.") for key in state_dict
    )

    if wrapped_checkpoint:
        model: nn.Module = SwinTiny(
            num_outputs=num_outputs,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
    else:
        model = swin_t(
            weights=None,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        model.head = nn.Linear(
            model.head.in_features,
            num_outputs,
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return model, effective_config, checkpoint


def read_manifest_items(
    manifest_path: Path,
    dataset_root: Path,
    limit: int | None,
) -> list[InputItem]:
    """Read labelled inputs while preserving manifest order."""
    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if not {"filename", "label"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Manifest must contain filename and label columns."
            )

        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    return [
        InputItem(
            path=dataset_root / row["filename"],
            relative_name=row["filename"],
            true_external_label=int(row["label"]),
        )
        for row in rows
    ]


def detections_from_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics result into project detections."""
    if result.boxes is None:
        return []

    names = result.names
    detections: list[Detection] = []

    for class_id, confidence, coordinates in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
        strict=True,
    ):
        class_index = int(class_id)

        detections.append(
            Detection(
                class_name=str(names[class_index]),
                confidence=float(confidence),
                x1=float(coordinates[0]),
                y1=float(coordinates[1]),
                x2=float(coordinates[2]),
                y2=float(coordinates[3]),
            )
        )

    return detections


def prepare_model_input(
    image: Image.Image,
    image_size: int,
) -> tuple[Image.Image, torch.Tensor]:
    """Apply deterministic resize, center crop, and normalization."""
    resize_size = round(image_size * 256 / 224)

    display_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]
    )

    display_image = display_transform(
        image.convert("RGB")
    )

    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                NORMALIZE_MEAN,
                NORMALIZE_STD,
            ),
        ]
    )(display_image)

    return display_image, tensor


def colorize_heatmap(
    heatmap: torch.Tensor,
    output_size: tuple[int, int],
) -> Image.Image:
    """Convert normalized Grad-CAM values into an RGB heatmap."""
    values = np.clip(
        heatmap.numpy(),
        0.0,
        1.0,
    )

    rgba = matplotlib.colormaps.get_cmap("turbo")(values)
    rgb = Image.fromarray(
        np.uint8(rgba[:, :, :3] * 255)
    )

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    )

    return rgb.resize(
        output_size,
        resample=resampling.BILINEAR,
    )


def overlay_heatmap(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float,
) -> Image.Image:
    """Overlay activated regions without tinting the entire image."""
    heatmap_image = colorize_heatmap(
        heatmap,
        image.size,
    )

    values = np.uint8(
        np.clip(
            heatmap.numpy(),
            0.0,
            1.0,
        )
        * 255
    )

    activation_mask = Image.fromarray(
        values,
        mode="L",
    )

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    )

    activation_mask = activation_mask.resize(
        image.size,
        resample=resampling.BILINEAR,
    )

    scaled_mask = activation_mask.point(
        lambda value: round(value * alpha)
    )

    return Image.composite(
        heatmap_image,
        image.convert("RGB"),
        scaled_mask,
    )


def safe_stem(index: int, name: str) -> str:
    """Create a stable and filesystem-safe artifact prefix."""
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(name).stem,
    ).strip("._")

    return f"{index:04d}_{cleaned or 'image'}"


def save_explanation_panel(
    path: Path,
    original: Image.Image,
    crop_box: tuple[int, int, int, int] | None,
    model_input: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    title: str,
) -> None:
    """Save a report-ready four-panel explanation."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 10),
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title(
        "Original image and YOLO crop"
        if crop_box
        else "Original image"
    )

    if crop_box is not None:
        left, top, right, bottom = crop_box

        axes[0, 0].add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
        )

    axes[0, 1].imshow(model_input)
    axes[0, 1].set_title(
        "Exact classifier input"
    )

    axes[1, 0].imshow(heatmap)
    axes[1, 0].set_title(
        "Grad-CAM (purple/blue: low → red: high)"
    )

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        "Activation-weighted overlay"
    )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        title,
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_detector_reject_panel(
    path: Path,
    original: Image.Image,
    title: str,
) -> None:
    """Explain a YOLO-gated reject where Swin never ran."""
    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    axis.imshow(original)
    axis.axis("off")
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    inputs = parser.add_mutually_exclusive_group(
        required=True
    )

    inputs.add_argument(
        "--image",
        type=Path,
        nargs="+",
    )
    inputs.add_argument(
        "--manifest",
        type=Path,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target-class",
        default="predicted",
        help=(
            "predicted, true, reject, external index -1..19, "
            "internal reject 20, or class name."
        ),
    )
    parser.add_argument(
        "--target-layer",
        default="stage3_last_norm1",
        choices=[
            "stage3_last_norm1",
            "last_block_norm1",
            "last_block_norm2",
            "final_norm",
        ],
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )
    parser.add_argument(
        "--yolo-device",
        default="0",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Override the checkpoint config's "
            "postprocessing threshold."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs"
            / "swin_gradcam"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError(
            "--limit must be positive."
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha must be between 0.0 and 1.0."
        )

    config_path = resolve_project_path(
        args.config
    )
    checkpoint_path = resolve_project_path(
        args.checkpoint
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    config = read_json(config_path)

    model, effective_config, checkpoint = (
        load_swin_checkpoint(
            checkpoint_path,
            config,
            device,
        )
    )

    data_config = effective_config.get(
        "data",
        {},
    )
    image_size = int(
        data_config.get("image_size", 224)
    )

    postprocessing = effective_config.get(
        "postprocessing",
        {},
    )

    confidence_threshold = (
        float(args.confidence_threshold)
        if args.confidence_threshold is not None
        else float(
            postprocessing.get("threshold", 0.0)
        )
    )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "The confidence threshold must be "
            "between 0.0 and 1.0."
        )

    if args.image:
        items = [
            InputItem(
                path=resolve_project_path(
                    image_path
                ),
                relative_name=image_path.name,
                true_external_label=None,
            )
            for image_path in args.image
        ]
    else:
        if args.dataset_root is None:
            raise ValueError(
                "--dataset-root is required "
                "when using --manifest."
            )

        items = read_manifest_items(
            resolve_project_path(args.manifest),
            resolve_project_path(
                args.dataset_root
            ),
            args.limit,
        )

    yolo_config = effective_config.get("yolo")
    use_yolo = (
        isinstance(yolo_config, dict)
        and not args.no_yolo
    )

    detector = None

    if use_yolo:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required by this config. "
                "Install requirements-yolo.txt or pass "
                "--no-yolo."
            ) from error

        detector = YOLO(
            str(
                yolo_config.get(
                    "model",
                    "yolov8n.pt",
                )
            )
        )

    output_dir = resolve_project_path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_layer = resolve_swin_target_layer(
        model,
        args.target_layer,
    )

    records: list[dict[str, Any]] = []

    with SwinGradCAM(
        model,
        target_layer,
    ) as explainer:
        for index, item in enumerate(
            items,
            start=1,
        ):
            if not item.path.is_file():
                raise FileNotFoundError(
                    "Input image was not found: "
                    f"{item.path}"
                )

            with Image.open(
                item.path
            ) as opened_image:
                original = opened_image.convert(
                    "RGB"
                )

            detection: Detection | None = None
            crop_box: (
                tuple[int, int, int, int] | None
            ) = None
            classifier_image = original

            if detector is not None:
                result = detector.predict(
                    source=str(item.path),
                    conf=float(
                        yolo_config.get(
                            "confidence_threshold",
                            0.25,
                        )
                    ),
                    device=args.yolo_device,
                    verbose=False,
                )[0]

                detection = (
                    select_largest_target_animal(
                        detections_from_result(
                            result
                        )
                    )
                )

                if detection is not None:
                    if bool(
                        yolo_config.get(
                            "square_crop",
                            True,
                        )
                    ):
                        crop_box = (
                            padded_square_crop_box(
                                detection,
                                original.width,
                                original.height,
                                float(
                                    yolo_config.get(
                                        "padding_fraction",
                                        0.10,
                                    )
                                ),
                            )
                        )
                    else:
                        crop_box = clamp_crop_box(
                            detection,
                            original.width,
                            original.height,
                        )

                if crop_box is None:
                    artifact_stem = safe_stem(
                        index,
                        item.relative_name,
                    )

                    panel_path = (
                        output_dir
                        / f"{artifact_stem}_panel.png"
                    )

                    save_detector_reject_panel(
                        panel_path,
                        original,
                        (
                            "YOLO found no valid cat/dog "
                            "crop; pipeline decision: reject"
                        ),
                    )

                    records.append(
                        {
                            "filename": (
                                item.relative_name
                            ),
                            "status": (
                                "detector_reject"
                            ),
                            "true_label": (
                                item.true_external_label
                            ),
                            "predicted_label": (
                                REJECT_EXTERNAL
                            ),
                            "panel": str(
                                panel_path
                            ),
                        }
                    )
                    continue

                classifier_image = original.crop(
                    crop_box
                )

            display_image, input_tensor = (
                prepare_model_input(
                    classifier_image,
                    image_size,
                )
            )

            requested_target = parse_target_class(
                args.target_class,
                item.true_external_label,
            )

            gradcam_result = explainer.generate(
                input_tensor.unsqueeze(0).to(
                    device
                ),
                requested_target,
            )

            raw_prediction = (
                gradcam_result.predicted_class
            )

            confidence = float(
                gradcam_result.probabilities[
                    raw_prediction
                ].item()
            )

            final_prediction = (
                REJECT_INTERNAL
                if confidence
                < confidence_threshold
                else raw_prediction
            )

            heatmap_image = colorize_heatmap(
                gradcam_result.heatmap,
                display_image.size,
            )

            overlay_image = overlay_heatmap(
                display_image,
                gradcam_result.heatmap,
                alpha=args.alpha,
            )

            artifact_stem = safe_stem(
                index,
                item.relative_name,
            )

            heatmap_path = (
                output_dir
                / f"{artifact_stem}_heatmap.png"
            )
            heatmap_values_path = (
                output_dir
                / f"{artifact_stem}_heatmap.npy"
            )
            overlay_path = (
                output_dir
                / f"{artifact_stem}_overlay.png"
            )
            panel_path = (
                output_dir
                / f"{artifact_stem}_panel.png"
            )
            metadata_path = (
                output_dir
                / f"{artifact_stem}_metadata.json"
            )

            heatmap_image.save(
                heatmap_path
            )

            np.save(
                heatmap_values_path,
                gradcam_result.heatmap.numpy(),
            )

            overlay_image.save(
                overlay_path
            )

            title = (
                "Raw prediction: "
                f"{class_name(raw_prediction)} "
                f"({confidence:.3f}) | "
                "final: "
                f"{class_name(final_prediction)} | "
                "Grad-CAM target: "
                f"{class_name(gradcam_result.target_class)}"
            )

            save_explanation_panel(
                panel_path,
                original,
                crop_box,
                display_image,
                heatmap_image,
                overlay_image,
                title,
            )

            metadata = {
                "filename": item.relative_name,
                "source_path": str(item.path),
                "checkpoint": str(
                    checkpoint_path
                ),
                "checkpoint_epoch": (
                    checkpoint.get("epoch")
                ),
                "target_layer": (
                    args.target_layer
                ),
                "target_internal_label": (
                    gradcam_result.target_class
                ),
                "target_external_label": (
                    internal_to_external(
                        gradcam_result.target_class
                    )
                ),
                "target_class_name": (
                    class_name(
                        gradcam_result.target_class
                    )
                ),
                "raw_prediction_internal": (
                    raw_prediction
                ),
                "raw_prediction_external": (
                    internal_to_external(
                        raw_prediction
                    )
                ),
                "raw_prediction_class_name": (
                    class_name(raw_prediction)
                ),
                "confidence": confidence,
                "confidence_threshold": (
                    confidence_threshold
                ),
                "final_prediction_external": (
                    internal_to_external(
                        final_prediction
                    )
                ),
                "final_prediction_class_name": (
                    class_name(
                        final_prediction
                    )
                ),
                "true_external_label": (
                    item.true_external_label
                ),
                "yolo_enabled": (
                    detector is not None
                ),
                "yolo_detected": (
                    detection is not None
                ),
                "yolo_confidence": (
                    detection.confidence
                    if detection is not None
                    else None
                ),
                "crop_box": (
                    list(crop_box)
                    if crop_box is not None
                    else None
                ),
                "heatmap": str(
                    heatmap_path
                ),
                "heatmap_values": str(
                    heatmap_values_path
                ),
                "overlay": str(
                    overlay_path
                ),
                "panel": str(
                    panel_path
                ),
            }

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

            records.append(
                {
                    "status": "explained",
                    **metadata,
                }
            )

    summary = {
        "config": str(config_path),
        "checkpoint": str(
            checkpoint_path
        ),
        "device": str(device),
        "target_layer": (
            args.target_layer
        ),
        "target_class_request": (
            args.target_class
        ),
        "images_requested": len(items),
        "images_explained": sum(
            record["status"] == "explained"
            for record in records
        ),
        "detector_rejects": sum(
            record["status"] == "detector_reject"
            for record in records
        ),
        "records": records,
    }

    summary_path = (
        output_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Grad-CAM outputs: "
        f"{output_dir.resolve()}"
    )
    print(
        f"Summary: "
        f"{summary_path.resolve()}"
    )


if __name__ == "__main__":
    main()
