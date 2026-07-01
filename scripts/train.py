"""PROBE training entrypoint.

Three-phase pipeline matching the paper (Sections 3.2–3.5):
  1. SPEM discovery — frozen ViT patch features → PCA + K-means → prototypes
  2. SSL pretraining — SimSiam + prompt consistency (InfoNCE) + DAPA (MMD)
  3. Detection training — FCOS-style head with Focal Loss + GIoU on source labels

Usage:
    # Full pipeline (all three phases)
    python scripts/train.py --config configs/probe_base.yaml --device cuda

    # Phase 3 only (from a Phase 2 checkpoint)
    python scripts/train.py --config configs/probe_base.yaml --phase 3 \
        --resume checkpoints/probe_final.pt --device cuda

    # With gradient accumulation (simulates larger batch)
    python scripts/train.py --config configs/probe_base.yaml --grad-accum 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(_PROJECT_ROOT)))
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "src"))

import timm
import yaml

from probe.data.road_damage import RoadDamageDataset
from probe.engine.self_training import (
    DomainAlignmentHead,
    SimSiamHeads,
    probe_pretrain_step,
)
from probe.engine.detection import (
    apply_nms,
    collect_detections,
    detection_loss,
    evaluate_map,
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


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def simsiam_transform(image_size: int = 512) -> T.Compose:
    """SimSiam-style two-view augmentation (paper Section 3.4)."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform(image_size: int = 512) -> T.Compose:
    """Simple resize + normalise transform for evaluation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def detection_collate(batch: list) -> tuple:
    """Collate variable-size detection targets (boxes/labels differ per image)."""
    images, targets = zip(*batch)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, 0)
    return images, targets


# ---------------------------------------------------------------------------
# Phase 1: SPEM prototype discovery (Section 3.2–3.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def discover_prototypes(
    dataset: RoadDamageDataset,
    vit_patch_embed: nn.Module,
    discovery: TargetPrototypeDiscovery,
    device: torch.device,
    image_size: int = 512,
    max_samples: int = 500,
) -> tuple[PrototypeState, torch.Tensor]:
    """Extract frozen ViT patch features from unlabeled target images, then
    run PCA + K-means to discover visual prototypes (SPEM).

    Returns the PrototypeState and the raw patch-feature tensor (for optional
    t-SNE visualisation).
    """
    indices = range(min(len(dataset), max_samples))
    transform = eval_transform(image_size)

    all_features = []
    print(f"SPEM: extracting patch features from {len(indices)} target images...")
    for idx in indices:
        img, _ = dataset[idx]
        tensor = transform(img).unsqueeze(0).to(device)
        patch_tokens = vit_patch_embed(tensor)
        if patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        features = patch_tokens.squeeze(0)  # [N_patches, D]
        all_features.append(features)

    patch_features = torch.cat(all_features, dim=0)
    print(f"  collected {patch_features.shape[0]:,} patch tokens × {patch_features.shape[1]}d")

    state = discovery.fit(patch_features)
    print(f"  discovered {state.centroids.shape[0]} prototypes in "
          f"{discovery.pca_dim}d PCA space")
    return state, patch_features


# ---------------------------------------------------------------------------
# Phase 2: SSL pretraining (Section 3.4)
# ---------------------------------------------------------------------------

def train_ssl_pretraining(
    model: PROBEModel,
    ssl_heads: SimSiamHeads,
    alignment_head: DomainAlignmentHead,
    prompt_projector: PromptProjector,
    prototype_state: PrototypeState,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """SimSiam + prompt consistency + DAPA pretraining loop.

    Returns the path to the final Phase 2 checkpoint.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Self-Supervised Pretraining")
    print("=" * 60)

    image_size = args.image_size

    # --- Datasets -----------------------------------------------------------
    source_dataset = RoadDamageDataset(
        cfg["data"]["source_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(image_size),
        image_size=image_size,
    )
    target_ssl_dataset = RoadDamageDataset(
        cfg["data"]["target_manifest"],
        cfg["data"]["image_root"],
        image_size=image_size,
    )

    batch_size = cfg["data"]["batch_size"]
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )
    target_loader = DataLoader(
        target_ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )

    # --- Optimiser ----------------------------------------------------------
    trainable = (
        list(ssl_heads.parameters())
        + list(alignment_head.parameters())
        + list(prompt_projector.parameters())
    )
    optimizer = AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    total_epochs = getattr(args, "epochs", None) or cfg["optim"]["pretrain_epochs"]
    grad_accum = args.grad_accum

    # AMP setup (same pattern as Phase 3)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    ssl_aug = simsiam_transform(image_size)
    prompt_weight = cfg["spem"]["prompt_weight"]
    dapa_weight = cfg["dapa"]["weight"]
    prompt_temperature = cfg["spem"]["prompt_temperature"]
    checkpoint_dir = Path(args.checkpoint_dir)

    effective_batch = batch_size * grad_accum
    print(f"Epochs: {total_epochs}  |  batch: {batch_size}  |  "
          f"grad_accum: {grad_accum}  |  effective batch: {effective_batch}")
    print(f"Loss weights: λ_ssl=1.0  λ_prompt={prompt_weight}  λ_dapa={dapa_weight}")

    history: dict[str, list[float]] = {"loss": [], "ssl": [], "prompt": [], "dapa": []}

    for epoch in range(total_epochs):
        ssl_heads.train()
        alignment_head.train()
        model.train()
        model.backbone.freeze_backbone()

        epoch_losses = {"loss": 0.0, "ssl": 0.0, "prompt": 0.0, "dapa": 0.0}
        optimizer.zero_grad(set_to_none=True)
        steps = 0

        for step, (source_batch, (target_imgs, _)) in enumerate(
            zip(source_loader, target_loader)
        ):
            source_images, _ = source_batch
            source_images = source_images.to(device)

            # Two independent SimSiam views of each target image
            target_view1 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)
            target_view2 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                metrics = probe_pretrain_step(
                    model, ssl_heads, alignment_head,
                    source_images, target_view1, target_view2,
                    prototype_state, optimizer,
                    prompt_weight=prompt_weight,
                    dapa_weight=dapa_weight,
                    prompt_temperature=prompt_temperature,
                    grad_accum=grad_accum,
                    scaler=scaler,
                )

            # Step after accumulation window
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                optimizer.step()
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in epoch_losses:
                epoch_losses[k] += metrics[k]
            steps = step + 1

            if step % args.log_interval == 0:
                print(
                    f"  epoch {epoch:3d} step {step:4d} | "
                    f"loss {metrics['loss']:.4f}  ssl {metrics['ssl']:.4f}  "
                    f"prompt {metrics['prompt']:.4f}  dapa {metrics['dapa']:.4f}"
                )

        # Epoch summary
        avg = {k: epoch_losses[k] / steps for k in epoch_losses}
        for k, v in avg.items():
            history[k].append(v)
        print(
            f"Epoch {epoch:3d} avg | "
            f"loss {avg['loss']:.4f}  ssl {avg['ssl']:.4f}  "
            f"prompt {avg['prompt']:.4f}  dapa {avg['dapa']:.4f}"
        )

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = checkpoint_dir / f"probe_epoch{epoch + 1:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ssl_heads": ssl_heads.state_dict(),
                "alignment_head": alignment_head.state_dict(),
                "prototype_state": prototype_state,
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  checkpoint → {ckpt_path}")

    # Final Phase 2 checkpoint
    final_path = checkpoint_dir / "probe_final.pt"
    torch.save({
        "epoch": total_epochs,
        "model": model.state_dict(),
        "ssl_heads": ssl_heads.state_dict(),
        "alignment_head": alignment_head.state_dict(),
        "prototype_state": prototype_state,
    }, final_path)
    print(f"Phase 2 complete.  Final checkpoint → {final_path}")

    # Clean up SSL heads to free GPU memory before Phase 3
    del ssl_heads, alignment_head, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return final_path


# ---------------------------------------------------------------------------
# Phase 3: Detection head training (Section 3.5)
# ---------------------------------------------------------------------------

def train_detection_head(
    model: PROBEModel,
    prototype_state: PrototypeState,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train the lightweight detection head on labeled source data.

    The backbone (PromptEnhancedViT + PromptProjector) stays frozen;
    only the LightweightDetectionHead parameters are optimised.
    """
    print("\n" + "=" * 60)
    print("Phase 3: Detection Head Training")
    print("=" * 60)

    image_size = args.image_size
    feature_size = image_size // 16  # ViT-Base/16 patch grid
    stride = float(image_size) / feature_size

    # Pre-compute grid of (x_ctr, y_ctr) locations in image pixels
    locations = generate_grid(feature_size, stride, device)
    print(f"Detection grid: {feature_size}×{feature_size}  |  "
          f"stride: {stride:.1f} px  |  image: {image_size}×{image_size}")

    # --- Datasets -----------------------------------------------------------
    source_dataset = RoadDamageDataset(
        cfg["data"]["source_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(image_size),
        image_size=image_size,
    )
    val_dataset = RoadDamageDataset(
        cfg["data"]["val_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(image_size),
        image_size=image_size,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        drop_last=False,
        collate_fn=detection_collate,
    )

    print(f"Source: {len(source_dataset)} images  |  Val: {len(val_dataset)} images")

    # --- Label sanity check & class distribution ----------------------------
    source_label_counts: dict[int, int] = {}
    source_unique = set()
    for _, target in source_dataset:
        for lbl in target["labels"].tolist():
            source_label_counts[lbl] = source_label_counts.get(lbl, 0) + 1
            source_unique.add(lbl)
    val_label_counts: dict[int, int] = {}
    val_unique = set()
    for _, target in val_dataset:
        for lbl in target["labels"].tolist():
            val_label_counts[lbl] = val_label_counts.get(lbl, 0) + 1
            val_unique.add(lbl)

    print(f"Source labels: {sorted(source_unique)}  |  Val labels: {sorted(val_unique)}")
    num_classes = cfg["detection"]["num_classes"]
    print(f"Per-class GT box counts ({num_classes} classes):")
    print(f"  {'Class':>6s}  {'Source':>8s}  {'Val':>8s}")
    for c in range(num_classes):
        sc = source_label_counts.get(c, 0)
        vc = val_label_counts.get(c, 0)
        flag = "  <-- MISSING" if (sc == 0 and vc == 0) else ""
        print(f"  {c:>6d}  {sc:>8d}  {vc:>8d}{flag}")
    extra_s = source_unique - set(range(num_classes))
    extra_v = val_unique - set(range(num_classes))
    if extra_s or extra_v:
        print(f"  WARNING: labels outside [0,{num_classes-1}] — source:{sorted(extra_s)} val:{sorted(extra_v)}")

    # --- Optimiser (detection head only) ------------------------------------
    det_cfg = cfg.get("detection_optim", {})
    optimizer = AdamW(
        model.detection_head.parameters(),
        lr=det_cfg.get("lr", 1e-4),
        weight_decay=det_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=det_cfg.get("epochs", 50),
        eta_min=det_cfg.get("lr_min", 1e-6),
    )

    # --- Settings -----------------------------------------------------------
    total_epochs = getattr(args, "det_epochs", None) or det_cfg.get("epochs", 50)
    focal_alpha = det_cfg.get("focal_alpha", 0.25)
    focal_gamma = det_cfg.get("focal_gamma", 2.0)
    box_weight = det_cfg.get("box_weight", 1.0)
    ctr_weight = det_cfg.get("ctr_weight", 1.0)
    score_threshold = det_cfg.get("score_threshold", 0.05)
    nms_threshold = det_cfg.get("nms_threshold", 0.5)
    val_interval = det_cfg.get("val_interval", 5)
    center_sampling_radius = det_cfg.get("center_sampling_radius", 1.5)
    checkpoint_dir = Path(args.checkpoint_dir)
    grad_accum = args.grad_accum
    effective_batch = cfg["data"]["batch_size"] * grad_accum

    # Freeze backbone
    model.backbone.freeze_backbone()
    for param in model.backbone.parameters():
        param.requires_grad = False

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and not args.no_amp
    # GradScaler only needed for fp16; bf16 has enough dynamic range
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    print(f"Epochs: {total_epochs}  |  batch: {cfg['data']['batch_size']}  |  "
          f"grad_accum: {grad_accum}  |  effective batch: {effective_batch}"
          f"  |  AMP: {amp_enabled} ({amp_dtype})"
          f"  |  GradScaler: {scaler.is_enabled()}")

    # torch.compile on A100 gives ~20% speedup (PyTorch >= 2.0)
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model.detection_head = torch.compile(
                model.detection_head, mode="reduce-overhead"
            )
            print("  torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")

    best_map = 0.0
    best_epoch = -1
    history_det: dict[str, list[float]] = {"cls": [], "box": [], "ctr": [], "total": [], "mAP": []}

    for epoch in range(total_epochs):
        model.detection_head.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = {"cls": 0.0, "box": 0.0, "ctr": 0.0, "total": 0.0}
        steps = 0

        for batch_idx, (images, targets) in enumerate(source_loader):
            images = images.to(device)

            # Forward through frozen backbone
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                _, patch_tokens, _ = model.encode(images, prototype_state)
                predictions = model.detection_head(patch_tokens)
                loss_dict = detection_loss(
                    predictions, targets, locations, stride,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    box_weight=box_weight,
                    ctr_weight=ctr_weight,
                    center_sampling_radius=center_sampling_radius,
                )
                scaled_loss = loss_dict["det_total"] / grad_accum

            # Backward (through GradScaler when fp16, direct when bf16/fp32)
            scaler.scale(scaled_loss).backward()

            # Step only after accumulation window
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)  # needed before clip_grad_norm for fp16
                torch.nn.utils.clip_grad_norm_(
                    model.detection_head.parameters(), max_norm=10.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[f"det_{k}"].item()
            steps += 1

            if steps % args.log_interval == 0:
                print(
                    f"  epoch {epoch:3d} step {steps:4d} | "
                    f"cls {loss_dict['det_cls'].item():.4f}  "
                    f"box {loss_dict['det_box'].item():.4f}  "
                    f"ctr {loss_dict['det_ctr'].item():.4f}  "
                    f"total {loss_dict['det_total'].item():.4f}"
                )

        scheduler.step()

        # Epoch summary
        avg = {k: epoch_losses[k] / max(steps, 1) for k in epoch_losses}
        for k, v in avg.items():
            history_det[k].append(v)
        print(
            f"Epoch {epoch:3d} avg | "
            f"cls {avg['cls']:.4f}  box {avg['box']:.4f}  "
            f"ctr {avg['ctr']:.4f}  total {avg['total']:.4f}  "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
            print("  evaluating mAP@0.5 ...")
            metrics = evaluate_map(
                model, val_dataset, prototype_state, device,
                locations, stride,
                num_classes=cfg["detection"]["num_classes"],
                iou_threshold=0.5,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                image_size=image_size,
            )
            mAP = metrics["mAP@0.5"]
            history_det["mAP"].append(mAP)
            class_aps = {k: v for k, v in metrics.items() if k.startswith("AP_cls_")}
            ap_parts = []
            for k, v in sorted(class_aps.items()):
                if v is None:
                    ap_parts.append(f"c{k.split('_')[-1]}=n/a")
                else:
                    ap_parts.append(f"c{k.split('_')[-1]}={v:.3f}")
            ap_str = "  ".join(ap_parts)
            print(f"  mAP@0.5: {mAP:.4f}  |  {ap_str}")

            if mAP > best_map:
                best_map = mAP
                best_epoch = epoch
                best_path = checkpoint_dir / "probe_det_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "detection_head": model.detection_head.state_dict(),
                    "prototype_state": prototype_state,
                    "optimizer": optimizer.state_dict(),
                    "mAP": mAP,
                    "class_aps": class_aps,
                }, best_path)
                print(f"  best model → {best_path}  (mAP={best_map:.4f})")

    # Final Phase 3 checkpoint
    final_path = checkpoint_dir / "probe_det_final.pt"
    torch.save({
        "epoch": total_epochs - 1,
        "model": model.state_dict(),
        "detection_head": model.detection_head.state_dict(),
        "prototype_state": prototype_state,
        "best_mAP": best_map,
        "best_epoch": best_epoch,
    }, final_path)
    print(f"\nPhase 3 complete.  Best mAP@0.5: {best_map:.4f}  (epoch {best_epoch})")
    print(f"Final checkpoint → {final_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROBE: Self-Supervised Visual Prompting for Cross-Domain "
                    "Road Damage Detection"
    )
    parser.add_argument("--config", default="configs/probe_base.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Input image size (paper: 512)")
    parser.add_argument("--spem-samples", type=int, default=500,
                        help="Max target images for SPEM discovery")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override Phase 2 pretrain epochs")
    parser.add_argument("--det-epochs", type=int, default=None,
                        help="Override Phase 3 detection epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (simulates larger batch)")
    parser.add_argument("--no-amp", action="store_true", default=False,
                        help="Disable AMP (useful for debugging or older GPUs)")
    parser.add_argument("--no-compile", action="store_true", default=False,
                        help="Disable torch.compile (A100: ~20 pct speedup when enabled)")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Run only a specific phase "
             "(3 = detection only, requires --resume)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a Phase 2 checkpoint (required for --phase 3)",
    )
    args = parser.parse_args()

    # --- Config -------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Build model --------------------------------------------------------
    print("Loading ViT backbone ...")
    vit = timm.create_model(
        cfg["backbone"]["name"], pretrained=True, img_size=args.image_size
    )
    vit.reset_classifier(0)

    discovery = TargetPrototypeDiscovery(
        pca_dim=cfg["spem"]["pca_dim"],
        num_prototypes=cfg["spem"]["num_prototypes"],
        kmeans_iters=cfg["spem"]["kmeans_iters"],
    )
    prompt_projector = PromptProjector(
        pca_dim=cfg["spem"]["pca_dim"],
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["spem"]["prompt_hidden_dim"],
    )
    backbone = PromptEnhancedViT(
        vit, prompt_projector,
        injection_layers=tuple(cfg["spem"]["injection_layers"]),
    )
    detection_head = LightweightDetectionHead(
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["detection"]["hidden_dim"],
        num_classes=cfg["detection"]["num_classes"],
        cls_prior=cfg["detection"].get("cls_prior", 0.01),
        head_depth=cfg["detection"].get("head_depth", 3),
    )
    model = PROBEModel(backbone, detection_head).to(device)

    # ======================================================================
    # Phase 3 only
    # ======================================================================
    if args.phase == 3:
        if args.resume is None:
            raise ValueError("--resume <checkpoint> is required for --phase 3")

        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        print(f"Loading pretrained backbone from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        # Load backbone weights only (skip detection head if present)
        model_state = ckpt["model"]
        filtered_state = {
            k: v for k, v in model_state.items()
            if not k.startswith("detection_head.")
        }
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded backbone (missing: {len(missing)}, "
              f"unexpected: {len(unexpected)})")

        prototype_state = ckpt.get("prototype_state")
        if prototype_state is None:
            raise ValueError("Checkpoint does not contain prototype_state")
        if isinstance(prototype_state, PrototypeState):
            prototype_state = PrototypeState(
                mean=prototype_state.mean.to(device),
                components=prototype_state.components.to(device),
                centroids=prototype_state.centroids.to(device),
            )
        else:
            # Legacy dict loading
            prototype_state = PrototypeState(
                mean=prototype_state["mean"].to(device),
                components=prototype_state["components"].to(device),
                centroids=prototype_state["centroids"].to(device),
            )

        train_detection_head(model, prototype_state, cfg, args, device)
        return

    # ======================================================================
    # Full pipeline: Phase 1 → Phase 2 → Phase 3
    # ======================================================================

    # --- Phase 1: SPEM discovery --------------------------------------------
    if args.phase is None or args.phase == 1:
        print("\n" + "=" * 60)
        print("Phase 1: SPEM Prototype Discovery")
        print("=" * 60)

        target_dataset = RoadDamageDataset(
            cfg["data"]["target_manifest"],
            cfg["data"]["image_root"],
        )
        prototype_state, _spem_features = discover_prototypes(
            target_dataset,
            vit_patch_embed=vit.patch_embed,
            discovery=discovery,
            device=device,
            image_size=args.image_size,
            max_samples=args.spem_samples,
        )
        prototype_state = PrototypeState(
            mean=prototype_state.mean.to(device),
            components=prototype_state.components.to(device),
            centroids=prototype_state.centroids.to(device),
        )
        del _spem_features

        if args.phase == 1:
            print("Phase 1 complete. (--phase 1: stopping here)")
            return

    # --- Phase 2: SSL pretraining -------------------------------------------
    if args.phase is None or args.phase == 2:
        ssl_heads = SimSiamHeads(
            embed_dim=cfg["backbone"]["embed_dim"],
            hidden_dim=cfg["ssl"]["hidden_dim"],
            out_dim=cfg["ssl"]["out_dim"],
        ).to(device)

        alignment_head = DomainAlignmentHead(
            embed_dim=cfg["backbone"]["embed_dim"],
            projection_dim=cfg["dapa"]["projection_dim"],
        ).to(device)

        final_path = train_ssl_pretraining(
            model, ssl_heads, alignment_head, prompt_projector,
            prototype_state, cfg, args, device,
        )

        if args.phase == 2:
            print("Phase 2 complete. (--phase 2: stopping here)")
            return

    # --- Phase 3: Detection training ----------------------------------------
    if args.phase is None or args.phase == 3:
        # Load the Phase 2 checkpoint to get the trained backbone + prototypes
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model_state = ckpt["model"]
        filtered_state = {
            k: v for k, v in model_state.items()
            if not k.startswith("detection_head.")
        }
        model.load_state_dict(filtered_state, strict=False)
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

        train_detection_head(model, prototype_state, cfg, args, device)


if __name__ == "__main__":
    main()
