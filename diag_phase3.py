"""Evaluate any Phase 3 checkpoint on a configured manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import timm
import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from probe.data.road_damage import RoadDamageDataset
from probe.engine.detection import evaluate_map, generate_grid
from probe.models import (
    LightweightDetectionHead,
    PROBEModel,
    PromptEnhancedViT,
    PromptProjector,
    PrototypeState,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_a100.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest-key",
        default="source_manifest",
        choices=("source_manifest", "source_val_manifest", "target_manifest", "val_manifest"),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--match-training-subset", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    vit = timm.create_model(
        cfg["backbone"]["name"], pretrained=False, img_size=args.image_size
    )
    vit.reset_classifier(0)
    prompt_projector = PromptProjector(
        cfg["spem"]["pca_dim"],
        cfg["backbone"]["embed_dim"],
        cfg["spem"]["prompt_hidden_dim"],
    )
    backbone = PromptEnhancedViT(
        vit,
        prompt_projector,
        injection_layers=tuple(cfg["spem"]["injection_layers"]),
        detection_layers=tuple(cfg["backbone"].get("detection_layers", ())),
    )
    det_cfg = cfg.get("detection_optim", {})
    use_centerness = det_cfg.get("ctr_weight", 0.0) > 0.0
    detection_head = LightweightDetectionHead(
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["detection"]["hidden_dim"],
        num_classes=cfg["detection"]["num_classes"],
        cls_prior=cfg["detection"].get("cls_prior", 0.01),
        head_depth=cfg["detection"].get("head_depth", 3),
        use_centerness=use_centerness,
        architecture=cfg["detection"].get(
            "architecture", "fcos" if use_centerness else "paper"
        ),
        paper_mid_dim=cfg["detection"].get("paper_mid_dim", 384),
        paper_neck_dim=cfg["detection"].get("paper_neck_dim", 128),
    )
    model = PROBEModel(backbone, detection_head).to(device)
    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)

    stored = checkpoint["prototype_state"]
    prototype_state = PrototypeState(
        stored["mean"].to(device) if isinstance(stored, dict) else stored.mean.to(device),
        stored["components"].to(device)
        if isinstance(stored, dict)
        else stored.components.to(device),
        stored["centroids"].to(device)
        if isinstance(stored, dict)
        else stored.centroids.to(device),
    )
    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    manifest = cfg["data"][args.manifest_key]
    dataset = RoadDamageDataset(
        manifest,
        cfg["data"]["image_root"],
        transform=transform,
        image_size=args.image_size,
        num_classes=cfg["detection"]["num_classes"],
    )
    if args.match_training_subset:
        if args.manifest_key != "source_manifest":
            raise ValueError("--match-training-subset requires --manifest-key source_manifest")
        sample_count = max(
            1,
            round(len(dataset) * float(cfg["detection"]["source_label_fraction"])),
        )
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        dataset = Subset(
            dataset,
            torch.randperm(len(dataset), generator=generator)[:sample_count].tolist(),
        )
    feature_size = args.image_size // 16
    metrics = evaluate_map(
        model,
        dataset,
        prototype_state,
        device,
        generate_grid(feature_size, args.image_size / feature_size, device),
        args.image_size / feature_size,
        num_classes=cfg["detection"]["num_classes"],
        score_threshold=det_cfg.get("score_threshold", 0.05),
        nms_threshold=det_cfg.get("nms_threshold", 0.5),
        image_size=args.image_size,
        max_samples=args.max_samples,
        use_centerness=use_centerness,
        box_mode=det_cfg.get(
            "box_mode", "ltrb" if use_centerness else "center_size"
        ),
    )
    print(f"manifest: {manifest} ({min(len(dataset), args.max_samples or len(dataset))} images)")
    print(f"mAP@50: {metrics['mAP@0.5'] * 100:.2f}%")
    print(f"mAP@[.5:.95]: {metrics['mAP@[.5:.95]'] * 100:.2f}%")
    for key, value in sorted(metrics.items()):
        if key.startswith("AP_cls_"):
            print(f"{key}: {'n/a' if value is None else f'{value * 100:.2f}%'}")


if __name__ == "__main__":
    main()
