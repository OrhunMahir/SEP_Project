#!/usr/bin/env python3
"""Audit an immutable manifest and save an auditable JSON summary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

# Run basic data checks without modifying the training dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Allow the manifest, image root, and output path to be overridden from the CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset" / "all")
    parser.add_argument("--content-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "data_audit.json")
    args = parser.parse_args()
    # Each manifest row contains one image path and its class label.
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    paths = [row["filename"] for row in rows]
    missing = [path for path in paths if not (args.image_root / path).is_file()]
    report = {
        "total_samples": len(rows),
        "class_counts_external": dict(sorted(Counter(int(row["label"]) for row in rows).items())),
        "missing_files": missing,
        "duplicate_manifest_paths": sorted({path for path, count in Counter(paths).items() if count > 1}),
        "content_hashes_requested": args.content_hashes,
    }

    if args.content_hashes:
        groups: dict[str, list[str]] = defaultdict(list)
        for relative_path in paths:
            image_path = args.image_root / relative_path
            if not image_path.is_file():
                continue
            digest = hashlib.sha256()
            with image_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            groups[digest.hexdigest()].append(relative_path)
        duplicate_groups = [
            sorted(group)
            for group in groups.values()
            if len(group) > 1
        ]
        report.update(
            {
                "content_hashes_computed": sum(len(group) for group in groups.values()),
                "unique_content_hashes": len(groups),
                "duplicate_content_groups": sorted(duplicate_groups),
            }
        )
    # Save only the audit output; never write into the source dataset.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
