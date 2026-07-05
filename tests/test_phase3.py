import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
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
from probe.models.detector import LightweightDetectionHead, PromptEnhancedViT
from probe.models.prompts import PromptProjector, PrototypeState


class _AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.value


_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "probe_train_script",
    Path(__file__).resolve().parents[1] / "scripts" / "train.py",
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MODULE = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN_MODULE)
_label_counts = _TRAIN_MODULE._label_counts


class DetectionDataTests(unittest.TestCase):
    def test_label_counts_support_subsets_without_loading_images(self) -> None:
        dataset = type(
            "ManifestOnlyDataset",
            (),
            {
                "samples": [
                    {"labels": [0, 1]},
                    {"labels": [1, 1]},
                    {"labels": []},
                ]
            },
        )()

        self.assertEqual(_label_counts(dataset), {0: 1, 1: 3})
        self.assertEqual(
            _label_counts(torch.utils.data.Subset(dataset, [0, 2])),
            {0: 1, 1: 1},
        )

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

    def test_unlabeled_mode_discards_manifest_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (32, 32)).save(root / "sample.jpg")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps({
                    "image": "sample.jpg",
                    "boxes": [[1, 2, 10, 12]],
                    "labels": [3],
                })
                + "\n",
                encoding="utf-8",
            )
            dataset = RoadDamageDataset(manifest, root, unlabeled=True)

            _, target = dataset[0]

            self.assertEqual(dataset.samples, [{"image": "sample.jpg"}])
            self.assertEqual(target["boxes"].numel(), 0)
            self.assertEqual(target["labels"].numel(), 0)


class DetectionModelTests(unittest.TestCase):
    def test_backbone_averages_requested_detection_layers(self) -> None:
        vit = nn.Module()
        vit.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        vit.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        vit.pos_embed = nn.Parameter(torch.zeros(1, 5, 8))
        vit.blocks = nn.ModuleList([_AddConstant(1.0), _AddConstant(3.0)])
        vit.norm = nn.Identity()
        projector = PromptProjector(pca_dim=2, embed_dim=8, hidden_dim=4)
        backbone = PromptEnhancedViT(
            vit,
            projector,
            injection_layers=(),
            detection_layers=(0, 1),
        )
        state = PrototypeState(
            mean=torch.zeros(8),
            components=torch.zeros(8, 2),
            centroids=torch.randn(3, 2),
        )
        images = torch.randn(2, 3, 8, 8)
        initial_patches = backbone.patchify(images)[:, 1:]

        _, fused_patches, _ = backbone(images, state)

        torch.testing.assert_close(fused_patches, initial_patches + 2.5)

    def test_backbone_returns_image_conditioned_prompts(self) -> None:
        vit = nn.Module()
        vit.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        vit.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        vit.pos_embed = nn.Parameter(torch.zeros(1, 5, 8))
        vit.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=8,
                nhead=2,
                batch_first=True,
                dropout=0.0,
            )
        ])
        vit.norm = nn.Identity()
        projector = PromptProjector(pca_dim=2, embed_dim=8, hidden_dim=4)
        backbone = PromptEnhancedViT(vit, projector, injection_layers=(0,))
        state = PrototypeState(
            mean=torch.zeros(8),
            components=torch.zeros(8, 2),
            centroids=torch.randn(3, 2),
        )

        _, _, returned_prompts = backbone(torch.randn(2, 3, 8, 8), state)
        raw_prompts = projector(state.centroids, batch_size=2)

        self.assertFalse(torch.allclose(returned_prompts, raw_prompts))
        self.assertFalse(torch.allclose(returned_prompts[0], returned_prompts[1]))

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
