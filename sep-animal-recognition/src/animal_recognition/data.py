"""Manifest-based datasets, transforms, and balanced sampling utilities."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .constants import REJECT_INTERNAL, external_to_internal

# Bu dosyada train ve validation verisini aynı kurallarla hazırlıyorum.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    """Path, training label, and manifest-relative path for one image."""

    path: Path
    label: int
    relative_path: str


def load_manifest(manifest_path: Path, image_root: Path) -> list[Sample]:
    """Load only images explicitly listed in a CSV manifest."""
    samples: list[Sample] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative_path = row["filename"]
            samples.append(
                Sample(
                    path=image_root / relative_path,
                    label=external_to_internal(int(row["label"])),
                    relative_path=relative_path,
                )
            )
    return samples


def load_split(split_path: Path, image_root: Path) -> list[Sample]:
    """Load a previously saved train or validation split."""
    return load_manifest(split_path, image_root)


def training_transform(
    image_size: int = 224,
    augmentation_config: dict[str, object] | None = None,
) -> transforms.Compose:
    """Return the controlled augmentations used only for training images."""
    config = augmentation_config or {}
    crop_scale = tuple(config.get("random_resized_crop_scale", (0.75, 1.0)))
    crop_ratio = tuple(config.get("crop_aspect_ratio", (0.75, 1.3333333333333333)))
    color_jitter_probability = float(config.get("color_jitter_probability", 1.0))
    random_grayscale_probability = float(config.get("random_grayscale_probability", 0.0))
    random_perspective_probability = float(config.get("random_perspective_probability", 0.0))
    random_erasing_probability = float(config.get("random_erasing_probability", 0.0))

    transform_steps: list[transforms.Transform] = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=crop_ratio),
        transforms.RandomHorizontalFlip(p=float(config.get("horizontal_flip_probability", 0.5))),
        transforms.RandomRotation(float(config.get("rotation_degrees", 10))),
    ]
    color_jitter = transforms.ColorJitter(
        brightness=float(config.get("brightness", 0.15)),
        contrast=float(config.get("contrast", 0.15)),
        saturation=float(config.get("saturation", 0.10)),
        hue=float(config.get("hue", 0.0)),
    )
    if color_jitter_probability >= 1.0:
        transform_steps.append(color_jitter)
    elif color_jitter_probability > 0.0:
        transform_steps.append(transforms.RandomApply([color_jitter], p=color_jitter_probability))
    if random_grayscale_probability > 0.0:
        transform_steps.append(transforms.RandomGrayscale(p=random_grayscale_probability))
    if random_perspective_probability > 0.0:
        transform_steps.append(
            transforms.RandomPerspective(
                distortion_scale=float(config.get("perspective_distortion", 0.10)),
                p=random_perspective_probability,
            )
        )
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    if random_erasing_probability > 0.0:
        transform_steps.append(
            transforms.RandomErasing(
                p=random_erasing_probability,
                scale=tuple(config.get("erasing_scale", (0.02, 0.10))),
            )
        )
    return transforms.Compose(transform_steps)


def evaluation_transform(image_size: int = 224) -> transforms.Compose:
    """Return deterministic transforms for validation and inference."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


class ManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Open an image and return its tensor, internal label, and relative path."""

    def __init__(self, samples: Sequence[Sample], transform: transforms.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, sample.relative_path


class BalancedEpochSampler(Sampler[int]):
    """Sample equally from each class so reject examples do not dominate training."""

    def __init__(self, samples: Sequence[Sample], seed: int = 42):
        self.indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            self.indices_by_label[sample.label].append(index)

        expected_labels = set(range(REJECT_INTERNAL + 1))
        if set(self.indices_by_label) != expected_labels:
            raise ValueError("Balanced sampling requires all 21 classes in the training split.")

        # Reject sınıfının baskınlaşmaması için her sınıftan eşit sayıda örnek alıyorum.
        self.samples_per_class = min(len(indices) for indices in self.indices_by_label.values())
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so a different deterministic sample can be drawn."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected: list[int] = []
        for label in sorted(self.indices_by_label):
            candidates = self.indices_by_label[label].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[: self.samples_per_class])
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return self.samples_per_class * len(self.indices_by_label)
