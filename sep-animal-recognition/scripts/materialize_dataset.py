#!/usr/bin/env python3
"""Materialize the fixed 5,432-image dataset from extracted public sources."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filename_index(root: Path | None) -> dict[str, list[Path]]:
    if root is None:
        return {}
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory was not found: {root}")
    result: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file():
            result[path.name].append(path)
    return result


def choose_source(
    index: dict[str, list[Path]],
    filename: str,
    preferred_component: str | None = None,
) -> Path:
    candidates = index.get(filename, [])
    if preferred_component is not None:
        preferred = [
            path
            for path in candidates
            if preferred_component.casefold() in {
                component.casefold() for component in path.parts
            }
        ]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            candidates = preferred
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one source file named {filename!r}; found {len(candidates)}."
        )
    return candidates[0]


def wikimedia_url_map(path: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (class_name, str(item["file"])): str(item["url"])
        for class_name, items in payload.items()
        for item in items
    }


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "SEP-animal-recognition/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "dataset" / "all")
    parser.add_argument("--oxford-root", type=Path)
    parser.add_argument("--stanford-root", type=Path)
    parser.add_argument("--animals10-root", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument(
        "--wikimedia-metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "wikimedia_sources.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that all manifest paths already exist.",
    )
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if len(rows) != 5432:
        raise ValueError(f"Expected 5,432 manifest rows, found {len(rows):,}.")

    if args.verify_only:
        missing = [
            row["filename"]
            for row in rows
            if not (args.output_root / row["filename"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images are missing; first: {missing[0]}"
            )
        print(f"PASS: all {len(rows):,} manifest images exist under {args.output_root}")
        return

    oxford = filename_index(args.oxford_root)
    stanford = filename_index(args.stanford_root)
    animals10 = filename_index(args.animals10_root)
    coco = filename_index(args.coco_root)
    wiki_urls = wikimedia_url_map(args.wikimedia_metadata)

    copied = 0
    downloaded = 0
    skipped = 0
    by_source: Counter[str] = Counter()

    for row in rows:
        relative = Path(row["filename"])
        destination = args.output_root / relative
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        name = relative.name

        if name.startswith("oxford__"):
            source_name = name.removeprefix("oxford__")
            source = choose_source(oxford, source_name)
            shutil.copy2(source, destination)
            source_kind = "oxford"
            copied += 1
        elif name.startswith("stanford__"):
            breed, source_name = name.removeprefix("stanford__").split("__", 1)
            source = choose_source(stanford, source_name, breed)
            shutil.copy2(source, destination)
            source_kind = "stanford_target"
            copied += 1
        elif name.startswith("reject_stanford__"):
            breed, source_name = name.removeprefix("reject_stanford__").split("__", 1)
            source = choose_source(stanford, source_name, breed)
            shutil.copy2(source, destination)
            source_kind = "stanford_reject"
            copied += 1
        elif name.startswith("reject_animals10__"):
            animal_class, source_name = name.removeprefix("reject_animals10__").split("__", 1)
            source = choose_source(animals10, source_name, animal_class)
            shutil.copy2(source, destination)
            source_kind = "animals10"
            copied += 1
        elif name.startswith("reject_coco__"):
            source_name = name.removeprefix("reject_coco__")
            source = choose_source(coco, source_name)
            shutil.copy2(source, destination)
            source_kind = "coco"
            copied += 1
        elif name.startswith("wikimedia__"):
            source_name = name.removeprefix("wikimedia__")
            key = (relative.parent.name, source_name)
            if key not in wiki_urls:
                raise KeyError(f"No Wikimedia URL recorded for {relative}")
            download(wiki_urls[key], destination)
            source_kind = "wikimedia"
            downloaded += 1
        else:
            raise ValueError(f"Unrecognized manifest source prefix: {relative}")

        by_source[source_kind] += 1

    missing = [
        row["filename"]
        for row in rows
        if not (args.output_root / row["filename"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"Materialization finished with {len(missing)} missing images.")

    print(
        json.dumps(
            {
                "manifest_rows": len(rows),
                "output_root": str(args.output_root),
                "copied": copied,
                "downloaded": downloaded,
                "skipped_existing": skipped,
                "new_files_by_source": dict(sorted(by_source.items())),
                "missing": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
