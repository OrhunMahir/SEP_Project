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


def padded_square_crop_box(
    detection: Detection,
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Return a square crop around a detection with proportional context padding."""
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction must be non-negative.")
    if detection.area <= 0.0 or image_width < 1 or image_height < 1:
        return None

    box_width = detection.x2 - detection.x1
    box_height = detection.y2 - detection.y1
    side = max(box_width, box_height) * (1.0 + 2.0 * padding_fraction)
    side = min(side, float(min(image_width, image_height)))
    if side <= 0.0:
        return None

    center_x = (detection.x1 + detection.x2) / 2.0
    center_y = (detection.y1 + detection.y2) / 2.0
    left = min(max(center_x - side / 2.0, 0.0), image_width - side)
    top = min(max(center_y - side / 2.0, 0.0), image_height - side)
    right = min(image_width, ceil(left + side))
    bottom = min(image_height, ceil(top + side))
    left = max(0, floor(left))
    top = max(0, floor(top))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
