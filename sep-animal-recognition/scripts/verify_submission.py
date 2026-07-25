#!/usr/bin/env python3
"""Run structural, manifest, environment, data, and checkpoint preflight checks."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import importlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
FINAL_CONFIGS = [
    "configs/custom_cnn_yolo_crop_padded_medium_aug_100ep.json",
    "configs/resnet18_yolo_crop_padded_100ep.json",
    "configs/efficientnet_b0_yolo_crop_padded_100ep.json",
    "configs/resnet18_pretrained_yolo_crop_padded_50ep.json",
    "configs/efficientnet_b0_pretrained_yolo_crop_padded_50ep.json",
    "configs/swin_tiny_pretrained_yolo_crop_padded_50ep.json",
]
EXPECTED_LABELS = set(range(20)) | {-1}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    required = [
        PACKAGE_ROOT / "Final_Report.pdf",
        PACKAGE_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data" / "labels.csv",
        PROJECT_ROOT / "data" / "metadata" / "image_sha256.csv",
        PROJECT_ROOT / "splits" / "train_seed42.csv",
        PROJECT_ROOT / "splits" / "val_seed42.csv",
        PROJECT_ROOT / "inference.py",
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        failures.append("Missing required files: " + ", ".join(missing_required))
    else:
        passes.append("Required submission files are present")

    unwanted_names = {".DS_Store", "__pycache__", ".idea", ".vscode"}
    unwanted = [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.name in unwanted_names
    ]
    if unwanted:
        failures.append("Unwanted generated/tool files found: " + ", ".join(map(str, unwanted)))
    else:
        passes.append("No generated cache directories or macOS metadata found")

    manifest_path = PROJECT_ROOT / "data" / "labels.csv"
    if manifest_path.is_file():
        rows = read_rows(manifest_path)
        paths = [row["filename"] for row in rows]
        labels = [int(row["label"]) for row in rows]
        if len(rows) != 5432:
            failures.append(f"Manifest has {len(rows):,} rows instead of 5,432")
        elif len(paths) != len(set(paths)):
            failures.append("Manifest contains duplicate relative paths")
        elif set(labels) != EXPECTED_LABELS:
            failures.append(f"Manifest label set is {sorted(set(labels))}")
        else:
            passes.append("Manifest has 5,432 unique paths and all 21 external labels")

        expected_counts_path = PROJECT_ROOT / "data" / "metadata" / "class_counts.json"
        if expected_counts_path.is_file():
            class_names = [
                "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
                "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
                "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
                "Golden_Retriever", "German_Shepherd", "Siberian_Husky",
                "Dalmatian", "Rottweiler", "reject",
            ]
            actual = Counter(class_names[20 if label == -1 else label] for label in labels)
            expected = json.loads(expected_counts_path.read_text(encoding="utf-8"))
            if dict(actual) != expected:
                failures.append("Manifest class counts do not match class_counts.json")
            else:
                passes.append("Manifest class counts match the recorded dataset composition")

        train_rows = read_rows(PROJECT_ROOT / "splits" / "train_seed42.csv")
        val_rows = read_rows(PROJECT_ROOT / "splits" / "val_seed42.csv")
        train_paths = {row["filename"] for row in train_rows}
        val_paths = {row["filename"] for row in val_rows}
        if train_paths & val_paths:
            failures.append("Train and validation split paths overlap")
        elif train_paths | val_paths != set(paths):
            failures.append("Fixed splits do not exactly partition data/labels.csv")
        else:
            passes.append(
                f"Seed-42 splits are disjoint and complete ({len(train_rows):,}/{len(val_rows):,})"
            )

    invalid_configs = []
    for path in (PROJECT_ROOT / "configs").rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid_configs.append(f"{path}: {error}")
    if invalid_configs:
        failures.append("Invalid JSON configs: " + "; ".join(invalid_configs))
    else:
        passes.append("All configuration files are valid JSON")

    data_paths = json.loads(
        (PROJECT_ROOT / "configs" / "data_paths.json").read_text(encoding="utf-8")
    )
    data_root = args.data_root or resolve_project_path(data_paths["train_image_root"])
    if data_root.is_dir() and manifest_path.is_file():
        missing_images = [
            row["filename"]
            for row in read_rows(manifest_path)
            if not (data_root / row["filename"]).is_file()
        ]
        if missing_images:
            failures.append(
                f"Data root is missing {len(missing_images)} images; first: {missing_images[0]}"
            )
        else:
            passes.append(f"All 5,432 manifest images exist under {data_root}")
    elif args.require_data:
        failures.append(f"Dataset root was not found: {data_root}")
    else:
        warnings.append(f"Dataset root is not populated yet: {data_root}")

    missing_checkpoints = []
    for relative in FINAL_CONFIGS:
        config = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        checkpoint = resolve_project_path(config["output_dir"]) / "best.pt"
        if not checkpoint.is_file():
            missing_checkpoints.append(str(checkpoint))
    if missing_checkpoints and args.require_checkpoints:
        failures.append("Missing final checkpoints: " + ", ".join(missing_checkpoints))
    elif missing_checkpoints:
        warnings.append(
            f"{len(missing_checkpoints)} final checkpoints are absent; run training before inference"
        )
    else:
        passes.append("All six final model checkpoints are present")

    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
    ]
    unavailable = []
    versions = {}
    for module_name, package_name in dependencies:
        try:
            module = importlib.import_module(module_name)
            versions[package_name] = getattr(module, "__version__", "installed")
        except ImportError:
            unavailable.append(package_name)
    if unavailable:
        warnings.append("Environment packages not installed: " + ", ".join(unavailable))
    else:
        passes.append("Runtime dependencies import successfully: " + json.dumps(versions))

    for message in passes:
        print(f"PASS: {message}")
    for message in warnings:
        print(f"WARN: {message}")
    for message in failures:
        print(f"FAIL: {message}")

    if failures:
        sys.exit(1)
    print(f"\nPreflight completed: {len(passes)} passed, {len(warnings)} warning(s), 0 failed.")


if __name__ == "__main__":
    main()
