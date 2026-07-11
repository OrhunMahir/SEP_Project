"""Tests for deterministic robustness corruptions."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from animal_recognition.robustness import CORRUPTIONS, apply_corruption


class RobustnessCorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
        self.image = Image.fromarray(pixels, mode="RGB")

    def test_every_corruption_preserves_image_shape(self) -> None:
        for corruption in CORRUPTIONS:
            with self.subTest(corruption=corruption):
                result = apply_corruption(self.image, corruption, 3, seed=42)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, self.image.size)

    def test_noise_is_reproducible_for_the_same_seed(self) -> None:
        first = np.asarray(apply_corruption(self.image, "gaussian_noise", 4, seed=7))
        second = np.asarray(apply_corruption(self.image, "gaussian_noise", 4, seed=7))
        np.testing.assert_array_equal(first, second)

    def test_invalid_severity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Severity"):
            apply_corruption(self.image, "contrast", 0)

    def test_unknown_corruption_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown corruption"):
            apply_corruption(self.image, "rain", 2)


if __name__ == "__main__":
    unittest.main()
