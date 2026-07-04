"""Combine labeled source data with class-balanced target pseudo-labels."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=600)
    args = parser.parse_args()

    train_images = args.output / "images" / "train"
    train_labels = args.output / "labels" / "train"
    link_tree(args.source / "images" / "train", train_images)
    link_tree(args.source / "labels" / "train", train_labels)
    link_tree(args.source / "images" / "val", args.output / "images" / "val")
    link_tree(args.source / "labels" / "val", args.output / "labels" / "val")

    candidates: dict[int, list[tuple[float, Path, str]]] = defaultdict(list)
    for path in args.predictions.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path}: expected class xywh confidence")
            candidates[int(fields[0])].append((float(fields[-1]), path, " ".join(fields[:-1])))

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
