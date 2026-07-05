"""Combine labeled source data with class-balanced target pseudo-labels."""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path


def prediction_rows(path: Path) -> list[tuple[int, tuple[float, float, float, float], float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}: expected class xywh confidence")
        rows.append((int(fields[0]), tuple(map(float, fields[1:5])), float(fields[5])))
    return rows


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    def corners(box):
        x, y, width, height = box
        return x - width / 2, y - height / 2, x + width / 2, y + height / 2

    ax1, ay1, ax2, ay2 = corners(first)
    bx1, by1, bx2, by2 = corners(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def consensus_rows(
    first: list[tuple[int, tuple[float, float, float, float], float]],
    second: list[tuple[int, tuple[float, float, float, float], float]],
    min_confidence: float,
    min_iou: float,
) -> list[tuple[int, tuple[float, float, float, float], float]]:
    agreed = []
    unused = set(range(len(second)))
    for class_id, box, confidence in sorted(first, key=lambda row: row[-1], reverse=True):
        matches = [
            (box_iou(box, other_box), index, other_box, other_confidence)
            for index, (other_class, other_box, other_confidence) in enumerate(second)
            if index in unused
            and class_id == other_class
            and other_confidence >= min_confidence
        ]
        if confidence < min_confidence or not matches:
            continue
        iou, index, other_box, other_confidence = max(matches)
        if iou < min_iou:
            continue
        unused.remove(index)
        merged_box = tuple((left + right) / 2 for left, right in zip(box, other_box))
        agreed.append((class_id, merged_box, math.sqrt(confidence * other_confidence)))
    return agreed


def link_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if not target.exists():
            os.symlink(path.resolve(), target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target-images", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--predictions-secondary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.6)
    args = parser.parse_args()

    train_images = args.output / "images" / "train"
    train_labels = args.output / "labels" / "train"
    link_tree(args.source / "images" / "train", train_images)
    link_tree(args.source / "labels" / "train", train_labels)
    link_tree(args.source / "images" / "val", args.output / "images" / "val")
    link_tree(args.source / "labels" / "val", args.output / "labels" / "val")

    candidates: dict[int, list[tuple[float, Path, str]]] = defaultdict(list)
    for path in args.predictions.glob("*.txt"):
        rows = prediction_rows(path)
        if args.predictions_secondary is not None:
            secondary = args.predictions_secondary / path.name
            if not secondary.exists():
                continue
            rows = consensus_rows(
                rows,
                prediction_rows(secondary),
                args.min_confidence,
                args.min_iou,
            )
        for class_id, box, confidence in rows:
            if confidence < args.min_confidence:
                continue
            label = f"{class_id} " + " ".join(f"{value:.8f}" for value in box)
            candidates[class_id].append((confidence, path, label))

    selected: dict[str, list[str]] = defaultdict(list)
    for class_id in sorted(candidates):
        class_candidates = sorted(candidates[class_id], reverse=True)
        for _, path, label in class_candidates[: args.per_class]:
            selected[path.stem].append(label)
        print(
            f"class {class_id}: selected "
            f"{min(len(class_candidates), args.per_class)}/{len(class_candidates)}"
        )

    for stem, labels in selected.items():
        image = next(args.target_images.glob(f"{stem}.*"))
        os.symlink(image.resolve(), train_images / image.name)
        (train_labels / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [longitudinal, transverse, alligator, pothole]\n",
        encoding="utf-8",
    )
    print(f"target images selected: {len(selected)}")


if __name__ == "__main__":
    main()
