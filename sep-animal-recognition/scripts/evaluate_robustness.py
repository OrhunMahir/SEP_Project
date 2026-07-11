#!/usr/bin/env python3
"""Benchmark a model or weighted ensemble against deterministic image corruptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from animal_recognition.data import NORMALIZE_MEAN, NORMALIZE_STD, Sample, load_split
from animal_recognition.metrics import classification_metrics
from animal_recognition.models import build_model
from animal_recognition.robustness import CORRUPTIONS, apply_corruption
from animal_recognition.thresholding import apply_confidence_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    """Loaded configuration and checkpoint metadata for one ensemble member."""

    config_path: Path
    checkpoint_path: Path
    config: dict
    samples: list[Sample]
    weight: float
    model: torch.nn.Module


class CorruptedManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Apply a fixed corruption after deterministic resize and center crop."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int,
        corruption: str | None,
        severity: int,
        seed: int,
    ) -> None:
        self.samples = list(samples)
        resize_size = int(round(image_size * 256 / 224))
        self.geometry = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            prepared = self.geometry(image.convert("RGB"))
        if self.corruption is not None:
            digest = hashlib.blake2b(
                f"{self.seed}:{sample.relative_path}".encode("utf-8"),
                digest_size=8,
            ).digest()
            sample_seed = int.from_bytes(digest, byteorder="big")
            prepared = apply_corruption(
                prepared,
                self.corruption,
                self.severity,
                seed=sample_seed,
            )
        return self.to_tensor(prepared), sample.label, sample.relative_path


def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str) -> Path:
    """Resolve project-relative paths without changing absolute paths."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_csv_values(value: str) -> list[str]:
    """Split a comma-separated CLI value and reject empty lists."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_weights(value: str | None, count: int) -> list[float]:
    """Return validated ensemble weights, defaulting to equal weighting."""
    if value is None:
        return [1.0 / count] * count
    weights = [float(item) for item in parse_csv_values(value)]
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, received {len(weights)}.")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Ensemble weights cannot be negative.")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("At least one ensemble weight must be positive.")
    return [weight / total for weight in weights]


def load_model_specs(
    config_paths: Sequence[Path],
    checkpoint_paths: Sequence[Path],
    weights: Sequence[float],
    max_samples: int | None,
    device: torch.device,
) -> list[ModelSpec]:
    """Load configurations and ensure every model evaluates the same manifest."""
    if len(config_paths) != len(checkpoint_paths):
        raise ValueError("Pass one --checkpoint for each --config.")

    data_paths = read_json(PROJECT_ROOT / "configs" / "data_paths.json")
    specs: list[ModelSpec] = []
    expected_rows: list[tuple[str, int]] | None = None
    for config_path, checkpoint_path, weight in zip(
        config_paths, checkpoint_paths, weights, strict=True
    ):
        fallback_config = read_json(config_path)
        checkpoint_path = resolve_project_path(str(checkpoint_path))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", fallback_config)
        if checkpoint.get("model_name") != config["model"]["name"]:
            raise ValueError(f"Model name mismatch in checkpoint: {checkpoint_path}")
        model = build_model(config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        data_config = config["data"]
        image_root = resolve_project_path(
            str(data_config.get("image_root", data_paths["train_image_root"]))
        )
        if not image_root.is_dir():
            raise FileNotFoundError(f"Configured image root was not found: {image_root}")
        samples = load_split(
            resolve_project_path(str(data_config["validation_split"])),
            image_root,
        )
        if max_samples is not None:
            samples = samples[:max_samples]
        rows = [(sample.relative_path, sample.label) for sample in samples]
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise ValueError("All ensemble members must use the same validation manifest order.")
        specs.append(
            ModelSpec(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=config,
                samples=samples,
                weight=weight,
                model=model,
            )
        )
    return specs


def collect_probabilities(
    spec: ModelSpec,
    device: torch.device,
    corruption: str | None,
    severity: int,
    num_workers: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collect probabilities for one model and corruption condition."""
    data_config = spec.config["data"]
    loader = DataLoader(
        CorruptedManifestDataset(
            spec.samples,
            int(data_config["image_size"]),
            corruption,
            severity,
            seed,
        ),
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probability_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, targets, relative_paths in loader:
            logits = spec.model(images.to(device, non_blocking=True))
            probability_batches.append(torch.softmax(logits, dim=1).cpu())
            target_batches.append(targets.cpu())
            paths.extend(relative_paths)
    return torch.cat(probability_batches), torch.cat(target_batches), paths


def evaluate_condition(
    specs: Sequence[ModelSpec],
    device: torch.device,
    corruption: str | None,
    severity: int,
    threshold: float,
    num_workers: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Evaluate one clean or corrupted condition across all ensemble members."""
    ensemble_probabilities: torch.Tensor | None = None
    reference_targets: torch.Tensor | None = None
    reference_paths: list[str] | None = None
    for spec in specs:
        probabilities, targets, paths = collect_probabilities(
            spec, device, corruption, severity, num_workers, seed
        )
        if reference_targets is None:
            reference_targets = targets
            reference_paths = paths
        elif not torch.equal(reference_targets, targets) or reference_paths != paths:
            raise ValueError("Ensemble predictions are not aligned to the same validation rows.")
        weighted = probabilities * spec.weight
        ensemble_probabilities = (
            weighted
            if ensemble_probabilities is None
            else ensemble_probabilities + weighted
        )

    if ensemble_probabilities is None or reference_targets is None:
        raise RuntimeError("No model probabilities were collected.")
    predictions, _ = apply_confidence_threshold(ensemble_probabilities, threshold)
    record: dict[str, float | int | str] = {
        "corruption": corruption or "clean",
        "severity": severity,
        "samples": len(reference_targets),
    }
    record.update(classification_metrics(reference_targets.tolist(), predictions.tolist()))
    return record


def add_clean_deltas(
    records: list[dict[str, float | int | str]],
) -> None:
    """Add percentage-point degradation relative to the clean baseline."""
    clean = records[0]
    for record in records:
        record["accuracy_drop_pp"] = 100.0 * (
            float(clean["accuracy"]) - float(record["accuracy"])
        )
        record["macro_f1_drop_pp"] = 100.0 * (
            float(clean["macro_f1"]) - float(record["macro_f1"])
        )


def write_records_csv(path: Path, records: Sequence[dict[str, float | int | str]]) -> None:
    """Write condition-level metrics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_plots(
    output_dir: Path,
    records: Sequence[dict[str, float | int | str]],
    corruptions: Sequence[str],
) -> None:
    """Render accuracy and macro-F1 degradation curves."""
    import matplotlib.pyplot as plt

    clean = records[0]
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for corruption in corruptions:
            rows = [row for row in records if row["corruption"] == corruption]
            axis.plot(
                [0, *[int(row["severity"]) for row in rows]],
                [float(clean[metric]), *[float(row[metric]) for row in rows]],
                marker="o",
                label=corruption.replace("_", " "),
            )
        axis.set_xlabel("Corruption severity")
        axis.set_ylabel(label)
        axis.set_xticks(range(0, 6))
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}_by_severity.png", dpi=180)
        plt.close(figure)


def write_markdown_report(
    path: Path,
    records: Sequence[dict[str, float | int | str]],
    threshold: float,
    model_specs: Sequence[ModelSpec],
) -> None:
    """Write a self-contained summary of the robustness benchmark."""
    clean = records[0]
    corrupted = records[1:]
    worst = max(corrupted, key=lambda row: float(row["macro_f1_drop_pp"]))
    highest_severity = max(int(row["severity"]) for row in corrupted)
    final_rows = [
        row for row in corrupted if int(row["severity"]) == highest_severity
    ]

    lines = [
        "# Model Robustness Benchmark",
        "",
        "## Evaluation Setup",
        "",
        f"- Validation samples: {int(clean['samples'])}",
        f"- Confidence threshold: `{threshold:.3f}`",
        f"- Ensemble members: {len(model_specs)}",
        "- Corruption severities: 1 (mild) to 5 (strong)",
        "",
        "| Model config | Weight |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{spec.config_path}` | {spec.weight:.3f} |" for spec in model_specs
    )
    lines.extend([
        "",
        "## Clean Baseline",
        "",
        "| Accuracy | Macro-F1 | Reject F1 | False accepts | False rejects |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {float(clean['accuracy']):.4f} | {float(clean['macro_f1']):.4f} | "
            f"{float(clean['reject_f1']):.4f} | {int(clean['false_accepts'])} | "
            f"{int(clean['false_rejects'])} |"
        ),
        "",
        f"## Highest Requested Severity Results (severity {highest_severity})",
        "",
        "| Corruption | Accuracy | Macro-F1 | Accuracy drop (pp) | Macro-F1 drop (pp) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in final_rows:
        lines.append(
            f"| {str(row['corruption']).replace('_', ' ')} | "
            f"{float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
            f"{float(row['accuracy_drop_pp']):.2f} | "
            f"{float(row['macro_f1_drop_pp']):.2f} |"
        )
    lines.extend([
        "",
        "## Key Finding",
        "",
        (
            f"The largest Macro-F1 degradation was observed for "
            f"**{str(worst['corruption']).replace('_', ' ')}** at severity "
            f"**{int(worst['severity'])}**, with a drop of "
            f"**{float(worst['macro_f1_drop_pp']):.2f} percentage points** "
            "relative to clean validation images."
        ),
        "",
        "## Curves",
        "",
        "![Accuracy by corruption severity](accuracy_by_severity.png)",
        "",
        "![Macro-F1 by corruption severity](macro_f1_by_severity.png)",
        "",
        "Detailed metrics for every condition are available in `robustness_results.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional comma-separated ensemble weights; defaults to equal weights.",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--corruptions",
        default=",".join(CORRUPTIONS),
        help="Comma-separated corruption names.",
    )
    parser.add_argument("--severities", default="1,2,3,4,5")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "robustness_benchmark",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    corruptions = parse_csv_values(args.corruptions)
    unknown = sorted(set(corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruptions: {', '.join(unknown)}")
    severities = [int(value) for value in parse_csv_values(args.severities)]
    if any(severity not in range(1, 6) for severity in severities):
        raise ValueError("--severities values must be between 1 and 5.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    weights = parse_weights(args.weights, len(args.config))
    specs = load_model_specs(
        args.config,
        args.checkpoint,
        weights,
        args.max_samples,
        device,
    )
    records = [
        evaluate_condition(
            specs,
            device,
            corruption=None,
            severity=0,
            threshold=args.threshold,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    ]
    for corruption in corruptions:
        for severity in severities:
            print(f"Evaluating {corruption}, severity {severity}...")
            records.append(
                evaluate_condition(
                    specs,
                    device,
                    corruption,
                    severity,
                    args.threshold,
                    args.num_workers,
                    args.seed,
                )
            )
    add_clean_deltas(records)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(output_dir / "robustness_results.csv", records)
    write_plots(output_dir, records, corruptions)
    write_markdown_report(
        output_dir / "model_robustness_report.md",
        records,
        args.threshold,
        specs,
    )
    summary = {
        "device": str(device),
        "threshold": args.threshold,
        "seed": args.seed,
        "corruptions": corruptions,
        "severities": severities,
        "models": [
            {
                "config": str(spec.config_path),
                "checkpoint": str(spec.checkpoint_path),
                "weight": spec.weight,
            }
            for spec in specs
        ],
        "records": records,
    }
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved robustness benchmark to: {output_dir}")


if __name__ == "__main__":
    main()
