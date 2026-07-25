#!/usr/bin/env python3
"""Generate or verify the SHA-256 list for the fixed dataset manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "image_sha256.csv",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    if args.verify:
        with args.output.open(newline="", encoding="utf-8") as handle:
            expected = {
                row["filename"]: row["sha256"]
                for row in csv.DictReader(handle)
            }
        failures = []
        for row in manifest_rows:
            relative = row["filename"]
            path = args.image_root / relative
            if not path.is_file() or sha256(path) != expected.get(relative):
                failures.append(relative)
        if failures:
            raise RuntimeError(
                f"Checksum verification failed for {len(failures)} files; first: {failures[0]}"
            )
        print(f"PASS: {len(manifest_rows):,} image checksums match")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sha256"])
        writer.writeheader()
        for row in manifest_rows:
            relative = row["filename"]
            path = args.image_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            writer.writerow({"filename": relative, "sha256": sha256(path)})
    print(f"Wrote {len(manifest_rows):,} checksums to {args.output}")


if __name__ == "__main__":
    main()
