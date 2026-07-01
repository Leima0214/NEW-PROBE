"""Detection training and inference for PROBE Phase 3.

FCOS-style dense prediction with [l, t, r, b] box encoding, Focal Loss,
GIoU regression loss, centerness BCE, and VOC 11-point mAP evaluation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import generalized_box_iou_loss


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def generate_grid(
    feature_size: int,
    stride: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    half = stride / 2.0
    shifts = torch.arange(0, feature_size, device=device, dtype=dtype) * stride + half
    shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
    locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)
    return locations


# ---------------------------------------------------------------------------
# Box encoding / decoding
# ---------------------------------------------------------------------------

def encode_boxes(
    gt_boxes: torch.Tensor,
    locations: torch.Tensor,
    stride: float,
    center_sampling_radius: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FCOS box encoding with center sampling (standard practice).

    Only grid locations within ``center_sampling_radius * stride`` of a box
    centre are eligible as positive samples.  This suppresses low-quality
    edge locations and stabilises training.
    """
    K = locations.shape[0]
    N = gt_boxes.shape[0]
    if N == 0:
        return (
            torch.zeros(K, 4, device=locations.device),
            torch.zeros(K, dtype=torch.bool, device=locations.device),
            torch.zeros(K, dtype=torch.int64, device=locations.device),  # safe fallback
        )

    x_ctr, y_ctr = locations[:, 0], locations[:, 1]
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    l = x_ctr[:, None] - x1[None, :]
    t = y_ctr[:, None] - y1[None, :]
    r = x2[None, :] - x_ctr[:, None]
    b = y2[None, :] - y_ctr[:, None]

    inside_box = (l >= 0.0) & (t >= 0.0) & (r >= 0.0) & (b >= 0.0)

    # Center sampling: restrict to grid cells near box centre
    # Uses L2 (circular) radius — standard FCOS practice
    if center_sampling_radius > 0:
        centre_x = (x1 + x2) * 0.5
        centre_y = (y1 + y2) * 0.5
        radius = center_sampling_radius * stride
        dx = x_ctr[:, None] - centre_x[None, :]
        dy = y_ctr[:, None] - centre_y[None, :]
        in_center = (dx * dx + dy * dy) < (radius * radius)
        inside = inside_box & in_center
    else:
        inside = inside_box

    areas = (x2 - x1) * (y2 - y1)
    inside_float = inside.float()
    huge = areas.max() + 1.0
    masked_areas = inside_float * areas[None, :] + (1.0 - inside_float) * huge
    assigned_idx = masked_areas.argmin(dim=1)
    mask = inside_float.sum(dim=1) > 0
    assigned_idx[~mask] = -1

    idx_safe = assigned_idx.clamp(min=0)
    assigned_boxes = gt_boxes[idx_safe]
    lt = torch.stack([
        x_ctr - assigned_boxes[:, 0], y_ctr - assigned_boxes[:, 1],
        assigned_boxes[:, 2] - x_ctr, assigned_boxes[:, 3] - y_ctr,
    ], dim=-1)
    targets = lt / stride
    targets[~mask] = 0.0
    return targets, mask, assigned_idx


def decode_boxes(
    box_preds: torch.Tensor,
    locations: torch.Tensor,
    stride: float,
    max_size: float = 512.0,
) -> torch.Tensor:
    lt_rb = torch.exp(box_preds) * stride
    x1 = locations[:, 0] - lt_rb[:, 0]
    y1 = locations[:, 1] - lt_rb[:, 1]
    x2 = locations[:, 0] + lt_rb[:, 2]
    y2 = locations[:, 1] + lt_rb[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes[:, 0].clamp_(min=0.0, max=max_size)
    boxes[:, 1].clamp_(min=0.0, max=max_size)
    boxes[:, 2].clamp_(min=0.0, max=max_size)
    boxes[:, 3].clamp_(min=0.0, max=max_size)
    return boxes


def compute_centerness_targets(
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    lt, rb = targets[:, :2], targets[:, 2:]
    lr = torch.cat([lt[:, 0:1], rb[:, 0:1]], dim=-1)
    tb = torch.cat([lt[:, 1:2], rb[:, 1:2]], dim=-1)
    left_right = lr.min(dim=-1).values / (lr.max(dim=-1).values + eps)
    top_bottom = tb.min(dim=-1).values / (tb.max(dim=-1).values + eps)
    centerness = torch.sqrt(left_right * top_bottom)
    centerness[~mask] = 0.0
    return centerness


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    p = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        num_pos = targets.sum()
        if num_pos > 0:
            return loss.sum() / num_pos
        return loss.sum() * 0.0
    elif reduction == "sum":
        return loss.sum()
    return loss


def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    return generalized_box_iou_loss(pred_boxes, gt_boxes).mean()


# ---------------------------------------------------------------------------
# Detection loss (with centerness)
# ---------------------------------------------------------------------------

def detection_loss(
    predictions: dict[str, torch.Tensor],
    targets: list[dict],
    locations: torch.Tensor,
    stride: float,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    box_weight: float = 1.0,
    ctr_weight: float = 1.0,
    center_sampling_radius: float = 1.5,
) -> dict[str, torch.Tensor]:
    cls_logits = predictions["class_logits"]
    box_preds = predictions["boxes"]
    ctr_logits = predictions["centerness"]

    B, C, H, W = cls_logits.shape
    K = H * W
    device = cls_logits.device

    cls_losses, box_losses, ctr_losses, ctr_neg_losses = [], [], [], []

    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)

        reg_target, pos_mask, assigned_idx = encode_boxes(
            gt_boxes, locations, stride, center_sampling_radius=center_sampling_radius
        )

        cls_target = torch.zeros(K, C, device=device)
        pos_idx = torch.where(pos_mask)[0]
        if pos_idx.numel() > 0:
            assigned_labels = gt_labels[assigned_idx[pos_idx]]
            cls_target[pos_idx, assigned_labels] = 1.0

        cls_pred = cls_logits[b].permute(1, 2, 0).reshape(K, C)
        box_pred = box_preds[b].permute(1, 2, 0).reshape(K, 4)
        ctr_pred = ctr_logits[b].permute(1, 2, 0).reshape(K)

        cls_loss = sigmoid_focal_loss(
            cls_pred, cls_target, alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )
        cls_losses.append(cls_loss)

        n_pos = pos_idx.numel()
        if n_pos > 0:
            pred_boxes_decoded = decode_boxes(box_pred[pos_idx], locations[pos_idx], stride)
            gt_boxes_assigned = gt_boxes[assigned_idx[pos_idx]]
            box_loss = giou_loss(pred_boxes_decoded, gt_boxes_assigned)
            box_losses.append(box_loss)

            ctr_target = compute_centerness_targets(reg_target, pos_mask)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred[pos_idx], ctr_target[pos_idx], reduction="mean"
            )
            ctr_losses.append(ctr_loss)

        # Centerness negative supervision: push background locations to 0
        # Applied to EVERY image (not just those with GT boxes) to prevent
        # centerness noise floor at sigmoid(0)=0.5 from polluting scores.
        neg_mask = (~pos_mask) if n_pos > 0 else torch.ones(K, dtype=torch.bool, device=device)
        ctr_loss_neg = F.binary_cross_entropy_with_logits(
            ctr_pred[neg_mask],
            torch.zeros(neg_mask.sum(), device=device),
            reduction="mean",
        )
        ctr_neg_losses.append(ctr_loss_neg)

    loss_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
    loss_box = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0, device=device)
    loss_ctr_pos = torch.stack(ctr_losses).mean() if ctr_losses else torch.tensor(0.0, device=device)
    loss_ctr_neg = torch.stack(ctr_neg_losses).mean() if ctr_neg_losses else torch.tensor(0.0, device=device)

    loss_ctr = loss_ctr_pos + 0.5 * loss_ctr_neg  # full centerness loss (pos + 0.5×neg)
    total = loss_cls + box_weight * loss_box + ctr_weight * loss_ctr
    return {"det_cls": loss_cls, "det_box": loss_box, "det_ctr": loss_ctr, "det_total": total}


# ---------------------------------------------------------------------------
# Inference (centerness-weighted scoring)
# ---------------------------------------------------------------------------

def collect_detections(
    predictions: dict[str, torch.Tensor],
    locations: torch.Tensor,
    stride: float,
    score_threshold: float = 0.05,
    max_detections: int = 196,
    image_size: float = 512.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_logits = predictions["class_logits"]
    box_preds = predictions["boxes"]
    ctr_logits = predictions["centerness"]

    C = cls_logits.shape[1]
    K = locations.shape[0]

    cls_logits = cls_logits[0].permute(1, 2, 0).reshape(K, C)
    box_preds = box_preds[0].permute(1, 2, 0).reshape(K, 4)
    ctr_preds = ctr_logits[0].permute(1, 2, 0).reshape(K, 1)

    # Centerness-weighted final score (standard FCOS: sqrt(class_score * centerness))
    cls_probs = cls_logits.sigmoid()
    ctr_probs = ctr_preds.sigmoid()
    scores = (cls_probs * ctr_probs).sqrt()
    max_scores, max_labels = scores.max(dim=1)

    keep = max_scores > score_threshold
    if not keep.any():
        return (
            torch.zeros(0, 4, device=cls_logits.device),
            torch.zeros(0, device=cls_logits.device),
            torch.zeros(0, dtype=torch.long, device=cls_logits.device),
        )

    scores = max_scores[keep]
    labels = max_labels[keep]
    box_preds_kept = box_preds[keep]
    locs = locations[keep]

    boxes = decode_boxes(box_preds_kept, locs, stride, max_size=image_size)

    if scores.numel() > max_detections:
        topk = scores.topk(max_detections).indices
        boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if valid.any():
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    else:
        return (
            torch.zeros(0, 4, device=cls_logits.device),
            torch.zeros(0, device=cls_logits.device),
            torch.zeros(0, dtype=torch.long, device=cls_logits.device),
        )

    return boxes, scores, labels


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = tv_batched_nms(boxes, scores, labels, iou_threshold)
    if len(keep) > max_detections:
        keep = keep[scores[keep].topk(max_detections).indices]
    return boxes[keep], scores[keep], labels[keep]


# ---------------------------------------------------------------------------
# mAP (VOC 2007 11-point)
# ---------------------------------------------------------------------------

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_voc_ap(recalls: list[float], precisions: list[float]) -> float:
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for t in torch.linspace(0, 1, 11).tolist():
        p_max = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t:
                p_max = max(p_max, p)
        ap += p_max / 11.0
    return ap


@torch.no_grad()
def evaluate_map(
    model,
    dataset,
    prototype_state,
    device: torch.device,
    locations: torch.Tensor,
    stride: float,
    num_classes: int = 5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    image_size: int = 512,
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    import torchvision.transforms as T

    model.eval()

    all_gt: dict[int, list[dict]] = {c: [] for c in range(num_classes)}
    all_det: dict[int, list[dict]] = {c: [] for c in range(num_classes)}

    indices = range(len(dataset))
    if max_samples is not None:
        indices = range(min(len(dataset), max_samples))

    for idx in indices:
        img, target = dataset[idx]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        for box, label in zip(gt_boxes, gt_labels):
            c = int(label.item())
            if c < num_classes:
                all_gt[c].append({"image_id": idx, "box": box, "matched": False})

        if isinstance(img, torch.Tensor):
            tensor = img.unsqueeze(0).to(device)
        else:
            _transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = _transform(img).unsqueeze(0).to(device)

        predictions = model.detect(tensor, prototype_state)
        det_boxes, det_scores, det_labels = collect_detections(
            predictions, locations, stride,
            score_threshold=score_threshold, image_size=image_size,
        )
        det_boxes, det_scores, det_labels = apply_nms(
            det_boxes, det_scores, det_labels, iou_threshold=nms_threshold,
        )

        for box, score, label in zip(det_boxes, det_scores, det_labels):
            c = int(label.item())
            if c < num_classes:
                all_det[c].append({
                    "image_id": idx, "confidence": score.item(), "box": box.cpu(),
                })

    aps = {}
    classes_with_gt = 0
    for c in range(num_classes):
        dets = all_det[c]
        gts = all_gt[c]
        dets.sort(key=lambda x: x["confidence"], reverse=True)
        for gt in gts:
            gt["matched"] = False

        total_gt = len(gts)

        # Skip classes that have no ground-truth in the validation set.
        # Including them as 0 AP would unfairly penalise mAP when the
        # dataset simply doesn't contain that damage category.
        if total_gt == 0:
            aps[f"AP_cls_{c}"] = None
            continue

        classes_with_gt += 1

        tp, fp = [], []
        for det in dets:
            best_iou, best_gt = 0.0, None
            for gt in gts:
                if gt["image_id"] != det["image_id"] or gt["matched"]:
                    continue
                iou = compute_iou(det["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_gt = iou, gt
            if best_iou >= iou_threshold and best_gt is not None:
                tp.append(1); fp.append(0)
                best_gt["matched"] = True
            else:
                tp.append(0); fp.append(1)

        tp_cum = torch.tensor(tp).cumsum(dim=0).tolist() if tp else []
        fp_cum = torch.tensor(fp).cumsum(dim=0).tolist() if fp else []
        recalls = [t / max(total_gt, 1) for t in tp_cum]
        precisions = [
            tp_cum[i] / max(tp_cum[i] + fp_cum[i], 1) for i in range(len(tp_cum))
        ]
        aps[f"AP_cls_{c}"] = compute_voc_ap(recalls, precisions)

    # mAP averaged only over classes that actually appear in the validation set
    valid_aps = [v for v in aps.values() if v is not None]
    aps["mAP@0.5"] = sum(valid_aps) / max(len(valid_aps), 1) if valid_aps else 0.0
    return aps
