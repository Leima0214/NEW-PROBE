from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class RoadDamageDataset(Dataset):
    """Road-damage detection dataset backed by a JSONL manifest.

    Each labeled line should contain:
    ``{"image": "x.jpg", "boxes": [[x1,y1,x2,y2]], "labels": [1]}``
    Unlabeled target-domain samples may omit ``boxes`` and ``labels``.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
        joint_transform: Callable | None = None,
        image_size: int | None = None,
        num_classes: int | None = None,
        unlabeled: bool = False,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_size = image_size
        self.num_classes = num_classes
        with Path(manifest).open("r", encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        self.samples = (
            [{"image": sample["image"]} for sample in samples]
            if unlabeled
            else samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.image_root / sample["image"]).convert("RGB")
        orig_w, orig_h = image.size

        # Scale boxes from original image size to target size
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])
        if len(boxes) != len(labels):
            raise ValueError(
                f"Sample {index} has {len(boxes)} boxes but {len(labels)} labels."
            )
        if self.image_size is not None and boxes:
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
            boxes = [[b[0] * scale_x, b[1] * scale_y,
                      b[2] * scale_x, b[3] * scale_y] for b in boxes]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        if self.num_classes is not None and labels_tensor.numel() > 0:
            min_label = int(labels_tensor.min())
            max_label = int(labels_tensor.max())
            if min_label < 0 or max_label >= self.num_classes:
                raise ValueError(
                    f"Sample {index} has labels outside [0, {self.num_classes - 1}]: "
                    f"min={min_label}, max={max_label}."
                )

        if self.joint_transform is not None:
            image, boxes_tensor = self.joint_transform(image, boxes_tensor)
        elif self.transform is not None:
            image = self.transform(image)

        if self.image_size is not None and boxes_tensor.numel() > 0:
            boxes_tensor[:, 0::2].clamp_(0, self.image_size)
            boxes_tensor[:, 1::2].clamp_(0, self.image_size)

        if boxes_tensor.numel() > 0:
            valid = (
                (boxes_tensor[:, 2] > boxes_tensor[:, 0])
                & (boxes_tensor[:, 3] > boxes_tensor[:, 1])
            )
            boxes_tensor = boxes_tensor[valid]
            labels_tensor = labels_tensor[valid]

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
        }
        return image, target
