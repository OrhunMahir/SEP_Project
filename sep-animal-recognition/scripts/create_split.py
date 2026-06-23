#!/usr/bin/env python3
"""Create a fixed stratified 80/20 train-validation split with seed 42."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1.")

    # Bu dosyada her sınıfın iki split'te de temsil edilmesini garanti ediyorum.
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    validation_counts: dict[int, int] = {}

    for label in sorted(by_label):
        group = by_label[label].copy()
        rng.shuffle(group)

        # Her sınıfın yaklaşık %20'sini validation'a ayırıyorum.
        validation_size = round(len(group) * args.validation_fraction)
        validation_size = min(max(validation_size, 1), len(group) - 1)
        validation_rows.extend(group[:validation_size])
        train_rows.extend(group[validation_size:])
        validation_counts[label] = validation_size

    # Bu CSV'ler tüm modellerde aynı kalacak; böylece karşılaştırma adil olacaktır.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_rows in (("train_seed42.csv", train_rows), ("val_seed42.csv", validation_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(split_rows)

    summary = {
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "total_samples": len(rows),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "train_class_counts_external": dict(sorted(Counter(int(row["label"]) for row in train_rows).items())),
        "validation_class_counts_external": dict(sorted(Counter(int(row["label"]) for row in validation_rows).items())),
        "validation_samples_per_class": dict(sorted(validation_counts.items())),
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
