"""Re-estimate detector BatchNorm statistics on unlabeled target images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        images = [
            args.image_root / json.loads(line)["image"]
            for line in handle
            if line.strip()
        ]
    wrapper = YOLO(args.model)
    model = wrapper.model.cuda().eval()
    batchnorms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    for module in batchnorms:
        module.train()
        module.momentum = 0.1

    letterbox = LetterBox(new_shape=(args.image_size, args.image_size))
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            batch = []
            for path in images[start : start + args.batch_size]:
                image = cv2.imread(str(path))
                image = letterbox(image=image)
                batch.append(
                    torch.from_numpy(image[:, :, ::-1].copy())
                    .permute(2, 0, 1)
                    .float()
                    .div_(255)
                )
            model(torch.stack(batch).cuda())
    wrapper.save(args.output)
    print(f"updated {len(batchnorms)} BatchNorm layers with {len(images)} target images")


if __name__ == "__main__":
    main()
