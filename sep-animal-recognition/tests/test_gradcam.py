"""Unit tests for the channels-last Swin Grad-CAM implementation."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if torch is not None:
    from animal_recognition.gradcam import (
        SwinGradCAM,
        resolve_swin_target_layer,
        unwrap_swin,
    )

    class ToyChannelsLastClassifier(nn.Module):
        """A deterministic miniature model with Swin-like spatial features."""

        def __init__(self) -> None:
            super().__init__()
            self.target = nn.Identity()
            self.head = nn.Linear(3, 2, bias=False)

            with torch.no_grad():
                self.head.weight.zero_()
                self.head.weight[0, 0] = 1.0
                self.head.weight[1, 1] = 1.0

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            features = self.target(
                inputs.permute(0, 2, 3, 1)
            )
            return self.head(
                features.mean(dim=(1, 2))
            )

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    class FakeSwin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Identity(),
                nn.Sequential(FakeBlock()),
                nn.Identity(),
                nn.Sequential(FakeBlock()),
                nn.Identity(),
                nn.Sequential(FakeBlock()),
                nn.Identity(),
                nn.Sequential(FakeBlock()),
            )
            self.norm = nn.Identity()

    class WrappedFakeSwin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = FakeSwin()


@unittest.skipUnless(
    torch is not None,
    "PyTorch is required for Grad-CAM tests.",
)
class SwinGradCAMTests(unittest.TestCase):
    def make_input(self) -> torch.Tensor:
        inputs = torch.zeros(1, 3, 4, 5)

        inputs[:, 0] = torch.arange(
            1,
            21,
            dtype=torch.float32,
        ).reshape(4, 5)

        inputs[:, 1] = (
            torch.arange(
                1,
                21,
                dtype=torch.float32,
            ).reshape(4, 5)
            * 0.25
        )

        return inputs

    def test_predicted_class_heatmap_is_normalized(self) -> None:
        model = ToyChannelsLastClassifier()

        with SwinGradCAM(
            model,
            model.target,
        ) as explainer:
            result = explainer.generate(
                self.make_input()
            )

        self.assertEqual(
            result.predicted_class,
            0,
        )
        self.assertEqual(
            result.target_class,
            0,
        )
        self.assertEqual(
            tuple(result.heatmap.shape),
            (4, 5),
        )
        self.assertGreaterEqual(
            float(result.heatmap.min()),
            0.0,
        )
        self.assertAlmostEqual(
            float(result.heatmap.max()),
            1.0,
        )
        self.assertAlmostEqual(
            float(result.probabilities.sum()),
            1.0,
            places=6,
        )
        self.assertEqual(
            len(model.target._forward_hooks),
            0,
        )

    def test_explicit_target_can_differ_from_prediction(self) -> None:
        model = ToyChannelsLastClassifier()

        with SwinGradCAM(
            model,
            model.target,
        ) as explainer:
            result = explainer.generate(
                self.make_input(),
                target_class=1,
            )

        self.assertEqual(
            result.predicted_class,
            0,
        )
        self.assertEqual(
            result.target_class,
            1,
        )
        self.assertAlmostEqual(
            float(result.heatmap.max()),
            1.0,
        )

    def test_invalid_target_is_rejected(self) -> None:
        model = ToyChannelsLastClassifier()

        with SwinGradCAM(
            model,
            model.target,
        ) as explainer:
            with self.assertRaises(ValueError):
                explainer.generate(
                    self.make_input(),
                    target_class=2,
                )

    def test_swin_layer_resolution_supports_wrappers(self) -> None:
        model = WrappedFakeSwin()

        self.assertIs(
            unwrap_swin(model),
            model.network,
        )

        self.assertIs(
            resolve_swin_target_layer(model),
            model.network.features[5][-1].norm1,
        )

        self.assertIs(
            resolve_swin_target_layer(
                model,
                "stage3_last_norm1",
            ),
            model.network.features[5][-1].norm1,
        )

        self.assertIs(
            resolve_swin_target_layer(
                model,
                "last_block_norm1",
            ),
            model.network.features[-1][-1].norm1,
        )

        self.assertIs(
            resolve_swin_target_layer(
                model,
                "last_block_norm2",
            ),
            model.network.features[-1][-1].norm2,
        )

        self.assertIs(
            resolve_swin_target_layer(
                model,
                "final_norm",
            ),
            model.network.norm,
        )


if __name__ == "__main__":
    unittest.main()
