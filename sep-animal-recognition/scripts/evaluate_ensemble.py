#!/usr/bin/env python3
"""Evaluate weighted softmax ensembles on the fixed internal validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from animal_recognition.data import ManifestDataset, evaluation_transform, load_split
from animal_recognition.metrics import (
    classification_metrics,
    write_confusion_matrix_csv,
    write_confusion_matrix_png,
)
from animal_recognition.models import build_model
from animal_recognition.thresholding import apply_confidence_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON configuration file."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths while keeping absolute paths unchanged."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def threshold_values(start: float, end: float, step: float) -> list[float]:
    """Create an inclusive, rounded threshold grid."""
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("Threshold start and end must satisfy 0.0 <= start <= end <= 1.0.")
    if step <= 0.0:
        raise ValueError("Threshold step must be positive.")

    values: list[float] = []
    current = start
    while current <= end + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def parse_float_list(text: str) -> list[float]:
    """Parse comma-separated floating point values."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one weight value is required.")
    return values


def ranking_key(
    record: dict[str, float | int],
    selection_metric: str,
) -> tuple[float, float, float, int, int, float, float]:
    """Rank ensemble candidates by the requested metric with stable tie-breakers."""
    if selection_metric == "accuracy":
        return (
            -float(record["accuracy"]),
            -float(record["macro_f1"]),
            -float(record["reject_f1"]),
            int(record["false_accepts"]),
            int(record["false_rejects"]),
            float(record["threshold"]),
            -float(record["model_0_weight"]),
        )
    return (
        -float(record["macro_f1"]),
        -float(record["reject_f1"]),
        -float(record["accuracy"]),
        int(record["false_accepts"]),
        int(record["false_rejects"]),
        float(record["threshold"]),
        -float(record["model_0_weight"]),
    )


def collect_validation_probabilities(
    config_path: Path,
    checkpoint_path: Path | None,
    device: torch.device,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict, Path]:
    """Load one checkpoint and return validation probabilities in split order."""
    config = read_json(config_path)
    output_dir = resolve_project_path(config["output_dir"])
    resolved_checkpoint = checkpoint_path or output_dir / "best.pt"
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {resolved_checkpoint}")

    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    if checkpoint.get("model_name") != checkpoint_config["model"]["name"]:
        raise ValueError(f"Checkpoint model name does not match: {resolved_checkpoint}")

    model = build_model(checkpoint_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    data_config = checkpoint_config["data"]
    image_root = resolve_project_path(
        str(data_config.get("image_root", data_paths["train_image_root"]))
    )
    if not image_root.is_dir():
        raise FileNotFoundError(f"Configured image root was not found: {image_root}")
    validation_samples = load_split(
        resolve_project_path(data_config["validation_split"]),
        image_root,
    )
    validation_loader = DataLoader(
        ManifestDataset(validation_samples, evaluation_transform(int(data_config["image_size"]))),
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probability_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    relative_paths: list[str] = []
    with torch.no_grad():
        for images, targets, batch_paths in validation_loader:
            logits = model(images.to(device, non_blocking=True))
            probability_batches.append(torch.softmax(logits, dim=1).cpu())
            target_batches.append(targets.cpu())
            relative_paths.extend(batch_paths)

    return (
        torch.cat(probability_batches),
        torch.cat(target_batches),
        relative_paths,
        checkpoint,
        resolved_checkpoint,
    )


def write_csv(path: Path, records: list[dict[str, float | int]]) -> None:
    """Write sweep rows in a format that can be opened in a spreadsheet."""
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Model config to ensemble. Pass exactly two configs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Optional checkpoint path for each config, in the same order.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--model-0-weights",
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated weights for the first model; second model gets 1-w.",
    )
    parser.add_argument(
        "--selection-metric",
        default="accuracy",
        choices=["macro_f1", "accuracy"],
        help="Metric used to choose the best ensemble and threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ensemble_resnet18_efficientnet_b0_padded",
    )
    args = parser.parse_args()

    if len(args.config) != 2:
        raise ValueError("This script currently expects exactly two --config arguments.")
    if args.checkpoint is not None and len(args.checkpoint) != len(args.config):
        raise ValueError("Pass either no --checkpoint values or one checkpoint per config.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint_paths = args.checkpoint or [None, None]
    model_outputs = [
        collect_validation_probabilities(config, checkpoint, device, args.num_workers)
        for config, checkpoint in zip(args.config, checkpoint_paths, strict=True)
    ]
    probabilities = [item[0] for item in model_outputs]
    targets = [item[1] for item in model_outputs]
    relative_paths = [item[2] for item in model_outputs]
    checkpoints = [item[3] for item in model_outputs]
    resolved_checkpoints = [item[4] for item in model_outputs]

    if not torch.equal(targets[0], targets[1]):
        raise ValueError("Validation targets differ between model configs.")
    if relative_paths[0] != relative_paths[1]:
        raise ValueError("Validation sample order differs between model configs.")

    target_list = targets[0].tolist()
    thresholds = threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    records: list[dict[str, float | int]] = []
    for model_0_weight in parse_float_list(args.model_0_weights):
        if not 0.0 <= model_0_weight <= 1.0:
            raise ValueError("Model weights must be between 0.0 and 1.0.")
        model_1_weight = 1.0 - model_0_weight
        ensemble_probabilities = (
            probabilities[0] * model_0_weight + probabilities[1] * model_1_weight
        )
        for threshold in thresholds:
            predictions, _ = apply_confidence_threshold(ensemble_probabilities, threshold)
            record: dict[str, float | int] = {
                "model_0_weight": round(model_0_weight, 6),
                "model_1_weight": round(model_1_weight, 6),
                "threshold": threshold,
            }
            record.update(classification_metrics(target_list, predictions.tolist()))
            records.append(record)

    best_record = min(records, key=lambda record: ranking_key(record, args.selection_metric))
    output_dir = resolve_project_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ensemble_sweep.csv", records)
    best_probabilities = (
        probabilities[0] * float(best_record["model_0_weight"])
        + probabilities[1] * float(best_record["model_1_weight"])
    )
    best_predictions, _ = apply_confidence_threshold(
        best_probabilities,
        float(best_record["threshold"]),
    )
    write_confusion_matrix_csv(
        output_dir / "confusion_matrix_best_ensemble.csv",
        target_list,
        best_predictions.tolist(),
    )
    write_confusion_matrix_png(
        output_dir / "confusion_matrix_best_ensemble.png",
        target_list,
        best_predictions.tolist(),
        (
            "Confusion Matrix: best ensemble "
            f"w0={float(best_record['model_0_weight']):.2f}, "
            f"tau={float(best_record['threshold']):.2f}"
        ),
    )
    result = {
        "device": str(device),
        "selection_metric": args.selection_metric,
        "validation_samples": len(target_list),
        "model_0": {
            "config": str(args.config[0]),
            "checkpoint": str(resolved_checkpoints[0]),
            "model_name": checkpoints[0]["model_name"],
            "checkpoint_epoch": int(checkpoints[0]["epoch"]),
        },
        "model_1": {
            "config": str(args.config[1]),
            "checkpoint": str(resolved_checkpoints[1]),
            "model_name": checkpoints[1]["model_name"],
            "checkpoint_epoch": int(checkpoints[1]["epoch"]),
        },
        "best_ensemble": best_record,
        "confusion_matrix_files": {
            "best_ensemble_csv": str(output_dir / "confusion_matrix_best_ensemble.csv"),
            "best_ensemble_png": str(output_dir / "confusion_matrix_best_ensemble.png"),
        },
        "candidates": records,
    }
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Selection metric: {args.selection_metric}")
    print(
        "Best ensemble: "
        f"model_0_weight={float(best_record['model_0_weight']):.2f} | "
        f"model_1_weight={float(best_record['model_1_weight']):.2f} | "
        f"threshold={float(best_record['threshold']):.2f} | "
        f"accuracy={float(best_record['accuracy']):.4f} | "
        f"macro_f1={float(best_record['macro_f1']):.4f} | "
        f"reject_f1={float(best_record['reject_f1']):.4f} | "
        f"false_accepts={int(best_record['false_accepts'])} | "
        f"false_rejects={int(best_record['false_rejects'])}"
    )
    print(f"Saved ensemble results to: {output_dir}")


if __name__ == "__main__":
    main()
