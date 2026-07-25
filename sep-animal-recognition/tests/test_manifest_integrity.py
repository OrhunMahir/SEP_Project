from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ManifestIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = read_rows(PROJECT_ROOT / "data" / "labels.csv")

    def test_manifest_shape(self) -> None:
        self.assertEqual(len(self.rows), 5432)
        paths = [row["filename"] for row in self.rows]
        self.assertEqual(len(paths), len(set(paths)))
        labels = {int(row["label"]) for row in self.rows}
        self.assertEqual(labels, set(range(20)) | {-1})

    def test_class_counts(self) -> None:
        class_names = [
            "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
            "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
            "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
            "Golden_Retriever", "German_Shepherd", "Siberian_Husky",
            "Dalmatian", "Rottweiler", "reject",
        ]
        actual = Counter(
            class_names[20 if int(row["label"]) == -1 else int(row["label"])]
            for row in self.rows
        )
        expected = json.loads(
            (PROJECT_ROOT / "data" / "metadata" / "class_counts.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(dict(actual), expected)

    def test_fixed_splits_partition_manifest(self) -> None:
        train = read_rows(PROJECT_ROOT / "splits" / "train_seed42.csv")
        validation = read_rows(PROJECT_ROOT / "splits" / "val_seed42.csv")
        train_paths = {row["filename"] for row in train}
        validation_paths = {row["filename"] for row in validation}
        manifest_paths = {row["filename"] for row in self.rows}
        self.assertFalse(train_paths & validation_paths)
        self.assertEqual(train_paths | validation_paths, manifest_paths)


if __name__ == "__main__":
    unittest.main()
