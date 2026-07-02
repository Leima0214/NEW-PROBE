import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe.data.road_damage import RoadDamageDataset
from probe.data.transforms import DetectionTrainTransform
from probe.engine.detection import (
    compute_interpolated_ap,
    detection_loss,
    encode_boxes,
    generate_grid,
    sigmoid_focal_loss,
)
from probe.models.detector import LightweightDetectionHead


class DetectionDataTests(unittest.TestCase):
    def test_horizontal_flip_updates_boxes(self) -> None:
        transform = DetectionTrainTransform(
            image_size=100,
            horizontal_flip_prob=1.0,
        )
        transform.color_jitter = lambda image: image
        image = Image.new("RGB", (100, 100))
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

        _, flipped = transform(image, boxes)

        torch.testing.assert_close(
            flipped,
            torch.tensor([[70.0, 20.0, 90.0, 40.0]]),
        )

    def test_empty_boxes_have_detection_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (32, 32)).save(root / "sample.jpg")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps({"image": "sample.jpg", "boxes": [], "labels": []})
                + "\n",
                encoding="utf-8",
            )
            dataset = RoadDamageDataset(manifest, root, image_size=32)

            _, target = dataset[0]

            self.assertEqual(tuple(target["boxes"].shape), (0, 4))
            self.assertEqual(tuple(target["labels"].shape), (0,))


class DetectionModelTests(unittest.TestCase):
    def test_paper_head_outputs_c_plus_four(self) -> None:
        head = LightweightDetectionHead(
            embed_dim=32,
            num_classes=4,
            architecture="paper",
            paper_mid_dim=16,
            paper_neck_dim=8,
        )
        predictions = head(torch.randn(2, 16, 32))

        self.assertEqual(tuple(predictions["class_logits"].shape), (2, 4, 4, 4))
        self.assertEqual(tuple(predictions["boxes"].shape), (2, 4, 4, 4))
        self.assertNotIn("centerness", predictions)

    def test_thin_box_gets_a_positive_location(self) -> None:
        locations = generate_grid(
            feature_size=4,
            stride=16.0,
            device=torch.device("cpu"),
        )
        # This two-pixel-high box contains no grid centre.
        boxes = torch.tensor([[20.0, 1.0, 44.0, 3.0]])

        _, mask, assigned = encode_boxes(
            boxes,
            locations,
            stride=16.0,
            guarantee_gt_match=True,
        )

        self.assertTrue(mask.any())
        self.assertIn(0, assigned[mask].tolist())

    def test_background_only_focal_loss_is_not_zero(self) -> None:
        logits = torch.zeros(8, 4)
        targets = torch.zeros_like(logits)

        loss = sigmoid_focal_loss(logits, targets)

        self.assertGreater(float(loss), 0.0)

    def test_paper_head_and_center_size_loss_backpropagate(self) -> None:
        head = LightweightDetectionHead(
            embed_dim=32,
            num_classes=4,
            architecture="paper",
            paper_mid_dim=16,
            paper_neck_dim=8,
        )
        predictions = head(torch.randn(2, 16, 32))
        locations = generate_grid(4, 16.0, torch.device("cpu"))
        targets = [
            {
                "boxes": torch.tensor([[18.0, 17.0, 42.0, 45.0]]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[1.0, 30.0, 63.0, 33.0]]),
                "labels": torch.tensor([2]),
            },
        ]

        losses = detection_loss(
            predictions,
            targets,
            locations,
            stride=16.0,
            ctr_weight=0.0,
            box_mode="center_size",
        )
        losses["det_total"].backward()

        self.assertTrue(torch.isfinite(losses["det_total"]))
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in head.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_continuous_ap_is_one_for_perfect_curve(self) -> None:
        ap = compute_interpolated_ap(
            recalls=[0.5, 1.0],
            precisions=[1.0, 1.0],
        )
        self.assertAlmostEqual(ap, 1.0)


if __name__ == "__main__":
    unittest.main()
