"""Export PROBE JSONL manifests as a symlinked Ultralytics dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def xyxy_to_yolo(box: list[float], width: int, height: int) -> tuple[float, ...]:
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def export_split(manifest: Path, image_root: Path, output: Path, split: str) -> None:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    for sample in samples:
        source = (image_root / sample["image"]).resolve()
        destination = image_dir / source.name
        if not destination.exists():
            os.symlink(source, destination)
        with Image.open(source) as image:
            width, height = image.size
        labels = sample.get("labels", [])
        boxes = sample.get("boxes", [])
        if len(labels) != len(boxes):
            raise ValueError(f"{source.name}: boxes/labels length mismatch")
        lines = []
        for label, box in zip(labels, boxes):
            x, y, w, h = xyxy_to_yolo(box, width, height)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{label} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    print(f"{split}: {len(samples)} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    assert xyxy_to_yolo([0, 0, 100, 50], 100, 50) == (0.5, 0.5, 1.0, 1.0)
    export_split(args.train_manifest, args.image_root, args.output, "train")
    export_split(args.val_manifest, args.image_root, args.output, "val")
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [longitudinal, transverse, alligator, pothole]\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
