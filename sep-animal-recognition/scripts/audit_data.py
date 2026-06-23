#!/usr/bin/env python3
"""Audit an immutable manifest and save an auditable JSON summary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

# Bu dosyada eğitim verisini değiştirmeden temel sağlık kontrolü yapıyorum.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Komut satırından farklı manifest, görüntü klasörü veya çıktı yolu verebilirim.
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset" / "labels.csv")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset" / "all")
    parser.add_argument("--content-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "data_audit.json")
    args = parser.parse_args()
    # Manifestteki her satır bir görüntü ve ona ait sınıf etiketidir.
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
        "note": "Use src/animal_recognition/data.py::audit_manifest(compute_sha256=True) in the configured ML environment for the optional content-hash pass.",
    }
    # Kaynak dataset'e yazmadan yalnızca denetim sonucunu kaydediyorum.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
