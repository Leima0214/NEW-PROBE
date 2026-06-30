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
        image_size: int | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_size = image_size
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.image_root / sample["image"]).convert("RGB")
        orig_w, orig_h = image.size

        # Scale boxes from original image size to target size
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])
        if self.image_size is not None and boxes:
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
            boxes = [[b[0] * scale_x, b[1] * scale_y,
                      b[2] * scale_x, b[3] * scale_y] for b in boxes]

        if self.transform is not None:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([index]),
        }
        return image, target
