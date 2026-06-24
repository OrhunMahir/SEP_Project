"""Small, dependency-free helpers for selecting a YOLO animal crop."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor


TARGET_ANIMAL_CLASSES = frozenset({"cat", "dog"})


@dataclass(frozen=True)
class Detection:
    """One object-detection result in pixel coordinates."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Return the non-negative bounding-box area."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def select_largest_target_animal(detections: list[Detection]) -> Detection | None:
    """Return the largest detected cat or dog, resolving ties by confidence."""
    candidates = [
        detection
        for detection in detections
        if detection.class_name.lower() in TARGET_ANIMAL_CLASSES and detection.area > 0.0
    ]
    return max(candidates, key=lambda detection: (detection.area, detection.confidence), default=None)


def clamp_crop_box(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp a detection to image bounds or return None for a degenerate crop."""
    left = max(0, floor(detection.x1))
    top = max(0, floor(detection.y1))
    right = min(image_width, ceil(detection.x2))
    bottom = min(image_height, ceil(detection.y2))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
