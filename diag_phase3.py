"""Self-diagnostic: evaluate Phase 3 model on its OWN training data.

If mAP on source (same-domain) is near 0, the detection head code has a bug.
If mAP on source is reasonable (>0.2), the issue is cross-domain features from Phase 2.
"""
import sys, yaml, torch, timm
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from probe.models import (PROBEModel, PromptEnhancedViT, PromptProjector,
                          LightweightDetectionHead, PrototypeState)
from probe.data.road_damage import RoadDamageDataset
from probe.engine.detection import (generate_grid, collect_detections,
                                    apply_nms, evaluate_map)
import torchvision.transforms as T

DEVICE = torch.device("cuda")
IMAGE_SIZE = 512
STRIDE = 16.0
FEATURE_SIZE = IMAGE_SIZE // 16
NUM_SAMPLES = 200  # evaluate on first 200 training images

# ── Load config & model ──────────────────────────────────────────
cfg = yaml.safe_load(open(ROOT / "configs/probe_a100.yaml"))

print("Loading ViT ...")
vit = timm.create_model(cfg["backbone"]["name"], pretrained=False, img_size=IMAGE_SIZE)
vit.reset_classifier(0)

print("Loading Phase 2 checkpoint ...")
# Try Phase 3 checkpoint first (trained head), fall back to Phase 2
det_ckpt_path = ROOT / "checkpoints/probe_det_best.pt"
if det_ckpt_path.exists():
    ckpt = torch.load(det_ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Loading Phase 3 detection checkpoint: {det_ckpt_path}")
else:
    ckpt = torch.load(ROOT / "checkpoints/probe_final.pt", map_location=DEVICE, weights_only=False)
    print("Loading Phase 2 checkpoint (detection head will be UNTRAINED!)")

prompt_projector = PromptProjector(50, 768, 256)
backbone = PromptEnhancedViT(
    vit,
    prompt_projector,
    injection_layers=(0, 6),
    detection_layers=tuple(cfg["backbone"].get("detection_layers", ())),
)

det_cfg = cfg.get("detection_optim", {})
use_ctr = det_cfg.get("ctr_weight", 1.0) > 0.0
det_head = LightweightDetectionHead(
    768, cfg["detection"]["hidden_dim"],
    cfg["detection"]["num_classes"],
    cls_prior=cfg["detection"].get("cls_prior", 0.01),
    use_centerness=use_ctr,
    architecture=cfg["detection"].get(
        "architecture", "fcos" if use_ctr else "paper"
    ),
    paper_mid_dim=cfg["detection"].get("paper_mid_dim", 384),
    paper_neck_dim=cfg["detection"].get("paper_neck_dim", 128),
)
model = PROBEModel(backbone, det_head).to(DEVICE)

# Load full model (backbone + detection head if in checkpoint)
# Strip _orig_mod. prefix from torch.compile-wrapped checkpoints
model_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
det_keys_ckpt = {k for k in model_state if "detection_head" in k}
det_keys_model = {k for k in model.state_dict() if "detection_head" in k}

print(f"  Detection head keys in checkpoint: {len(det_keys_ckpt)}")
print(f"  Detection head keys in model:      {len(det_keys_model)}")

# Check shape mismatches
mismatches = []
for k in sorted(det_keys_ckpt & det_keys_model):
    if model_state[k].shape != model.state_dict()[k].shape:
        mismatches.append(f"    {k}: ckpt={list(model_state[k].shape)} vs model={list(model.state_dict()[k].shape)}")
if mismatches:
    print("  SHAPE MISMATCHES:")
    for m in mismatches[:10]:
        print(m)
    if len(mismatches) > 10:
        print(f"    ... and {len(mismatches)-10} more")

missing, unexpected = model.load_state_dict(model_state, strict=False)
has_head = any("detection_head" in k for k in model_state)
print(f"  Loaded (missing: {len(missing)}, unexpected: {len(unexpected)})"
      f"  |  head_in_ckpt: {has_head}")

# Load prototype state
ps = ckpt["prototype_state"]
if isinstance(ps, dict):
    # legacy dict format
    prototype_state = PrototypeState(ps["mean"].to(DEVICE), ps["components"].to(DEVICE),
                                     ps["centroids"].to(DEVICE))
else:
    prototype_state = PrototypeState(ps.mean.to(DEVICE), ps.components.to(DEVICE),
                                     ps.centroids.to(DEVICE))

model.eval()
locations = generate_grid(FEATURE_SIZE, STRIDE, DEVICE)

# ── Evaluate on SOURCE training data ─────────────────────────────
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"\nEvaluating on source training data ({NUM_SAMPLES} images) ...")
src_dataset = RoadDamageDataset(
    cfg["data"]["source_manifest"],
    cfg["data"]["image_root"],
    transform=transform, image_size=IMAGE_SIZE,
)

metrics = evaluate_map(
    model, src_dataset, prototype_state, DEVICE,
    locations, STRIDE,
    num_classes=cfg["detection"]["num_classes"],
    score_threshold=det_cfg.get("score_threshold", 0.005),
    nms_threshold=det_cfg.get("nms_threshold", 0.5),
    image_size=IMAGE_SIZE,
    max_samples=NUM_SAMPLES,
    use_centerness=use_ctr,
    box_mode=det_cfg.get("box_mode", "ltrb" if use_ctr else "center_size"),
)

print(f"\n{'='*60}")
print(f"SOURCE-DOMAIN mAP@0.5: {metrics['mAP@0.5']:.4f}")
for k, v in sorted(metrics.items()):
    if k.startswith("AP_cls_"):
        label = "n/a" if v is None else f"{v:.4f}"
        print(f"  {k}: {label}")
print(f"{'='*60}")

if metrics["mAP@0.5"] < 0.05:
    print("\n⚠️  mAP < 0.05 on TRAINING data → DETECTION HEAD CODE HAS A BUG")
    print("   The model cannot detect objects even in its own training domain.")
elif metrics["mAP@0.5"] < 0.20:
    print("\n⚠️  mAP 0.05-0.20 on training data → Detection head is weak but functional")
    print("   Check label assignment, loss weights, or learning rate.")
else:
    print("\nSource-domain fitting is functional (mAP > 0.20).")
    print("This verifies checkpoint loading and basic fitting only; it does not")
    print("prove paper alignment or rule out data, metric, and transfer defects.")
