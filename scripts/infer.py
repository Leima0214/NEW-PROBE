"""PROBE inference entrypoint.

Load a trained PROBE checkpoint and run detection on images from a manifest.

Usage:
    python scripts/infer.py --config configs/probe_base.yaml \\
        --checkpoint checkpoints/probe_det_best.pt \\
        --manifest data/val.jsonl --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from tqdm import tqdm

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(_PROJECT_ROOT)))
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "src"))

import timm
import yaml

from probe.data.road_damage import RoadDamageDataset
from probe.engine.detection import (
    apply_nms,
    collect_detections,
    generate_grid,
)
from probe.models import (
    LightweightDetectionHead,
    PROBEModel,
    PromptEnhancedViT,
    PromptProjector,
    PrototypeState,
    TargetPrototypeDiscovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROBE inference on road-damage images"
    )
    parser.add_argument("--config", default="configs/probe_base.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a Phase 3 detection checkpoint (*_det_best.pt)")
    parser.add_argument("--manifest", default=None,
                        help="JSONL manifest of images to run inference on "
                             "(defaults to val_manifest from config)")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON file for detection results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load config --------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # --- Build model --------------------------------------------------------
    print("Loading ViT backbone ...")
    vit = timm.create_model(
        cfg["backbone"]["name"], pretrained=False, img_size=args.image_size
    )
    vit.reset_classifier(0)

    prompt_projector = PromptProjector(
        pca_dim=cfg["spem"]["pca_dim"],
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["spem"]["prompt_hidden_dim"],
    )
    backbone = PromptEnhancedViT(
        vit, prompt_projector,
        injection_layers=tuple(cfg["spem"]["injection_layers"]),
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
            "architecture",
            "fcos" if use_centerness else "paper",
        ),
        paper_mid_dim=cfg["detection"].get("paper_mid_dim", 384),
        paper_neck_dim=cfg["detection"].get("paper_neck_dim", 128),
    )
    model = PROBEModel(backbone, detection_head).to(device)

    # --- Load checkpoint ----------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = {
        key.replace("_orig_mod.", ""): value
        for key, value in ckpt["model"].items()
    }
    model.load_state_dict(model_state, strict=True)

    prototype_state = ckpt["prototype_state"]
    if isinstance(prototype_state, PrototypeState):
        prototype_state = PrototypeState(
            mean=prototype_state.mean.to(device),
            components=prototype_state.components.to(device),
            centroids=prototype_state.centroids.to(device),
        )
    else:
        prototype_state = PrototypeState(
            mean=prototype_state["mean"].to(device),
            components=prototype_state["components"].to(device),
            centroids=prototype_state["centroids"].to(device),
        )

    model.eval()

    # --- Grid ---------------------------------------------------------------
    feature_size = args.image_size // 16
    stride = float(args.image_size) / feature_size
    locations = generate_grid(feature_size, stride, device)

    # --- Dataset ------------------------------------------------------------
    manifest = args.manifest or cfg["data"]["val_manifest"]
    dataset = RoadDamageDataset(
        manifest,
        cfg["data"]["image_root"],
        transform=T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]),
        image_size=args.image_size,
        num_classes=cfg["detection"]["num_classes"],
    )
    print(f"Running inference on {len(dataset)} images ...")

    # --- Inference loop -----------------------------------------------------
    results = []
    for idx in tqdm(range(len(dataset))):
        img, _target = dataset[idx]
        tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model.detect(tensor, prototype_state)

        boxes, scores, labels = collect_detections(
            predictions, locations, stride,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            image_size=args.image_size,
            use_centerness=use_centerness,
            box_mode=det_cfg.get(
                "box_mode",
                "ltrb" if use_centerness else "center_size",
            ),
        )
        boxes, scores, labels = apply_nms(
            boxes, scores, labels,
            iou_threshold=args.nms_threshold,
            max_detections=args.max_detections,
        )

        results.append({
            "image_id": idx,
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "labels": labels.cpu().tolist(),
        })

    # --- Save ---------------------------------------------------------------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}  ({len(results)} images)")


if __name__ == "__main__":
    main()
