"""Deterministic image corruptions for robustness evaluation."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

CORRUPTIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
    "occlusion",
)
MIN_SEVERITY = 1
MAX_SEVERITY = 5

_NOISE_SIGMA = (8.0, 16.0, 24.0, 32.0, 40.0)
_BLUR_RADIUS = (0.8, 1.5, 2.2, 3.0, 4.0)
_BRIGHTNESS_FACTOR = (0.85, 0.70, 0.55, 0.40, 0.25)
_CONTRAST_FACTOR = (0.85, 0.70, 0.55, 0.40, 0.25)
_JPEG_QUALITY = (80, 60, 40, 25, 10)
_OCCLUSION_SIDE_FRACTION = (0.15, 0.25, 0.35, 0.45, 0.55)


def validate_corruption(corruption: str, severity: int) -> None:
    """Validate a corruption name and its one-based severity."""
    if corruption not in CORRUPTIONS:
        choices = ", ".join(CORRUPTIONS)
        raise ValueError(f"Unknown corruption '{corruption}'. Choose from: {choices}.")
    if not MIN_SEVERITY <= severity <= MAX_SEVERITY:
        raise ValueError(
            f"Severity must be between {MIN_SEVERITY} and {MAX_SEVERITY}, received {severity}."
        )


def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
    *,
    seed: int = 0,
) -> Image.Image:
    """Apply one deterministic corruption while preserving image dimensions."""
    validate_corruption(corruption, severity)
    rgb_image = image.convert("RGB")
    level = severity - 1

    if corruption == "gaussian_noise":
        pixels = np.asarray(rgb_image, dtype=np.float32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, _NOISE_SIGMA[level], size=pixels.shape)
        corrupted = np.clip(pixels + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(corrupted, mode="RGB")

    if corruption == "gaussian_blur":
        return rgb_image.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS[level]))

    if corruption == "brightness":
        return ImageEnhance.Brightness(rgb_image).enhance(_BRIGHTNESS_FACTOR[level])

    if corruption == "contrast":
        return ImageEnhance.Contrast(rgb_image).enhance(_CONTRAST_FACTOR[level])

    if corruption == "jpeg_compression":
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=_JPEG_QUALITY[level])
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

    side_fraction = _OCCLUSION_SIDE_FRACTION[level]
    width, height = rgb_image.size
    occlusion_width = max(1, round(width * side_fraction))
    occlusion_height = max(1, round(height * side_fraction))
    left = (width - occlusion_width) // 2
    top = (height - occlusion_height) // 2
    result = rgb_image.copy()
    neutral_gray = round(sum(ImageStat.Stat(rgb_image).mean) / 3.0)
    result.paste(
        (neutral_gray, neutral_gray, neutral_gray),
        (left, top, left + occlusion_width, top + occlusion_height),
    )
    return result
