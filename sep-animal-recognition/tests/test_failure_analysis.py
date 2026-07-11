"""Tests for prediction failure analysis."""

from __future__ import annotations

import unittest

from animal_recognition.failure_analysis import (
    analysis_summary,
    confidence_bin_rows,
    confusion_pair_rows,
    failure_rows,
    parse_prediction_records,
    per_class_failure_rows,
)


def sample_rows() -> list[dict[str, str]]:
    return [
        {"filename": "a.jpg", "label": "0", "prediction": "0", "confidence": "0.90"},
        {"filename": "b.jpg", "label": "0", "prediction": "1", "confidence": "0.85"},
        {"filename": "c.jpg", "label": "-1", "prediction": "2", "confidence": "0.80"},
        {"filename": "d.jpg", "label": "3", "prediction": "-1", "confidence": "0.40"},
        {"filename": "e.jpg", "label": "0", "prediction": "1", "confidence": "0.75"},
    ]


class FailureAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = parse_prediction_records(sample_rows())

    def test_failure_types_and_confidence_order(self) -> None:
        rows = failure_rows(self.records)
        self.assertEqual(
            [row["failure_type"] for row in rows],
            ["class_confusion", "false_accept", "class_confusion", "false_reject"],
        )
        self.assertEqual(rows[0]["filename"], "b.jpg")

    def test_confusion_pairs_are_aggregated(self) -> None:
        rows = confusion_pair_rows(self.records)
        self.assertEqual(rows[0]["true_class"], "Abyssinian")
        self.assertEqual(rows[0]["predicted_class"], "Bengal")
        self.assertEqual(rows[0]["count"], 2)
        self.assertAlmostEqual(rows[0]["mean_confidence"], 0.80)

    def test_per_class_rows_put_hardest_class_first(self) -> None:
        rows = per_class_failure_rows(self.records)
        self.assertEqual(rows[0]["accuracy"], 0.0)
        reject = next(row for row in rows if row["class_name"] == "reject")
        self.assertEqual(reject["false_accepts"], 1)
        abyssinian = next(row for row in rows if row["class_name"] == "Abyssinian")
        self.assertAlmostEqual(abyssinian["accuracy"], 1 / 3)

    def test_summary_counts_failure_modes_and_calibration(self) -> None:
        bins = confidence_bin_rows(self.records, num_bins=5)
        summary = analysis_summary(self.records, bins, high_confidence_threshold=0.8)
        self.assertEqual(summary["failures"], 4)
        self.assertEqual(summary["class_confusions"], 2)
        self.assertEqual(summary["false_accepts"], 1)
        self.assertEqual(summary["false_rejects"], 1)
        self.assertEqual(summary["high_confidence_failures"], 2)
        self.assertGreaterEqual(summary["expected_calibration_error"], 0.0)

    def test_duplicate_filenames_are_rejected(self) -> None:
        rows = sample_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_prediction_records(rows)


if __name__ == "__main__":
    unittest.main()
