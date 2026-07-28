from __future__ import annotations

import unittest

from animal_recognition.checkpointing import resolve_checkpoint_configs


class CheckpointConfigCompatibilityTests(unittest.TestCase):
    def test_saved_model_and_current_data_config_are_combined(self) -> None:
        checkpoint = {
            "config": {
                "model": {"name": "resnet18", "pretrained": True},
                "data": {"image_root": "runs/historical_model_cache"},
            }
        }
        submitted_config = {
            "model": {"name": "resnet18", "pretrained": True},
            "data": {
                "image_root": "runs/yolo_crops_final",
                "validation_split": "splits/val_seed42.csv",
            },
        }

        model_config, data_config = resolve_checkpoint_configs(
            checkpoint,
            submitted_config,
        )

        self.assertEqual(model_config, checkpoint["config"]["model"])
        self.assertEqual(data_config, submitted_config["data"])
        self.assertNotEqual(
            data_config["image_root"],
            checkpoint["config"]["data"]["image_root"],
        )

    def test_checkpoint_without_embedded_config_uses_submitted_config(self) -> None:
        submitted_config = {
            "model": {"name": "custom_cnn"},
            "data": {"image_root": "dataset/all"},
        }

        model_config, data_config = resolve_checkpoint_configs(
            {},
            submitted_config,
        )

        self.assertEqual(model_config, submitted_config["model"])
        self.assertEqual(data_config, submitted_config["data"])


if __name__ == "__main__":
    unittest.main()
