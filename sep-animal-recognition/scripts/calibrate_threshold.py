#!/usr/bin/env python3
"""Select a reject-confidence threshold on the fixed internal validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from animal_recognition.data import ManifestDataset, evaluation_transform, load_split
from animal_recognition.metrics import classification_metrics
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


def collect_validation_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic validation inference once and retain softmax probabilities."""
    model.eval()
    probability_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            probability_batches.append(torch.softmax(logits, dim=1).cpu())
            target_batches.append(targets.cpu())
    return torch.cat(probability_batches), torch.cat(target_batches)


def ranking_key(
    record: dict[str, float | int],
    selection_metric: str,
) -> tuple[float, float, float, int, int, float]:
    """Rank threshold candidates by the requested metric with stable tie-breakers."""
    if selection_metric == "accuracy":
        return (
            -float(record["accuracy"]),
            -float(record["macro_f1"]),
            -float(record["reject_f1"]),
            int(record["false_accepts"]),
            int(record["false_rejects"]),
            float(record["threshold"]),
        )
    return (
        -float(record["macro_f1"]),
        -float(record["reject_f1"]),
        -float(record["accuracy"]),
        int(record["false_accepts"]),
        int(record["false_rejects"]),
        float(record["threshold"]),
    )


def selection_rule_text(selection_metric: str) -> str:
    """Describe the threshold ranking rule stored in the calibration summary."""
    if selection_metric == "accuracy":
        return "accuracy, then macro_f1, reject_f1, false_accepts, false_rejects"
    return "macro_f1, then reject_f1, accuracy, false_accepts, false_rejects"


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
        default=PROJECT_ROOT / "configs" / "custom_cnn_baseline.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--selection-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy"],
        help="Metric used to choose the best threshold from the sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    output_dir = resolve_project_path(config["output_dir"])
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    if checkpoint.get("model_name") != checkpoint_config["model"]["name"]:
        raise ValueError("Checkpoint model name does not match its saved configuration.")

    model = build_model(checkpoint_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    probabilities, targets = collect_validation_probabilities(model, validation_loader, device)
    target_list = targets.tolist()
    records: list[dict[str, float | int]] = []
    for threshold in threshold_values(args.threshold_start, args.threshold_end, args.threshold_step):
        predictions, _ = apply_confidence_threshold(probabilities, threshold)
        record: dict[str, float | int] = {"threshold": threshold}
        record.update(classification_metrics(target_list, predictions.tolist()))
        records.append(record)

    best_record = min(records, key=lambda record: ranking_key(record, args.selection_metric))
    baseline_record = records[0]
    default_calibration_name = (
        "threshold_calibration"
        if args.selection_metric == "macro_f1"
        else f"threshold_calibration_{args.selection_metric}"
    )
    calibration_dir = args.output_dir or output_dir / default_calibration_name
    calibration_dir.mkdir(parents=True, exist_ok=True)
    write_csv(calibration_dir / "threshold_sweep.csv", records)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "validation_samples": len(target_list),
        "selection_metric": args.selection_metric,
        "selection_rule": selection_rule_text(args.selection_metric),
        "baseline_without_threshold": baseline_record,
        "best_threshold": best_record,
        "candidates": records,
    }
    (calibration_dir / "threshold_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Baseline macro-F1: {float(baseline_record['macro_f1']):.4f}")
    print(
        "Best threshold: "
        f"{float(best_record['threshold']):.2f} | "
        f"accuracy={float(best_record['accuracy']):.4f} | "
        f"macro_f1={float(best_record['macro_f1']):.4f} | "
        f"reject_f1={float(best_record['reject_f1']):.4f} | "
        f"false_accepts={int(best_record['false_accepts'])} | "
        f"false_rejects={int(best_record['false_rejects'])}"
    )
    print(f"Saved threshold results to: {calibration_dir}")


if __name__ == "__main__":
    main()
