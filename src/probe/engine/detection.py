"""Detection training, inference, and evaluation for PROBE Phase 3.

The paper-aligned path uses C+4 dense prediction with center-size boxes,
Focal Loss, GIoU regression, class-wise NMS, and continuous AP metrics.
The earlier FCOS-style ltrb/centerness path remains available for ablations.
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
    guarantee_gt_match: bool = False,
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

    # Thin road cracks can fall entirely between ViT patch centres. For the
    # centre-size box parameterisation, reserve the nearest free grid location
    # so every GT contributes classification and GIoU supervision.
    if guarantee_gt_match:
        represented = set(assigned_idx[mask].tolist())
        claimed = set(torch.where(mask)[0].tolist())
        centre_x = (x1 + x2) * 0.5
        centre_y = (y1 + y2) * 0.5
        centres = torch.stack([centre_x, centre_y], dim=-1)
        distances = torch.cdist(centres, locations)
        for gt_idx in range(N):
            if gt_idx in represented:
                continue
            candidates = distances[gt_idx].argsort()
            location_idx = next(
                (int(idx) for idx in candidates.tolist() if int(idx) not in claimed),
                int(candidates[0]),
            )
            assigned_idx[location_idx] = gt_idx
            mask[location_idx] = True
            claimed.add(location_idx)
            represented.add(gt_idx)

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
    box_mode: str = "center_size",
) -> torch.Tensor:
    if box_mode == "ltrb":
        lt_rb = F.softplus(box_preds) * stride
        x1 = locations[:, 0] - lt_rb[:, 0]
        y1 = locations[:, 1] - lt_rb[:, 1]
        x2 = locations[:, 0] + lt_rb[:, 2]
        y2 = locations[:, 1] + lt_rb[:, 3]
    elif box_mode == "center_size":
        centre_offsets = torch.tanh(box_preds[:, :2]) * stride
        centres = locations + centre_offsets
        sizes = F.softplus(box_preds[:, 2:]) * stride
        half_sizes = sizes * 0.5
        x1 = centres[:, 0] - half_sizes[:, 0]
        y1 = centres[:, 1] - half_sizes[:, 1]
        x2 = centres[:, 0] + half_sizes[:, 0]
        y2 = centres[:, 1] + half_sizes[:, 1]
    else:
        raise ValueError(f"Unknown box_mode: {box_mode}")

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
        return loss.sum() / num_pos.clamp(min=1.0)
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
    box_mode: str = "center_size",
) -> dict[str, torch.Tensor]:
    cls_logits = predictions["class_logits"]
    box_preds = predictions["boxes"]

    B, C, H, W = cls_logits.shape
    K = H * W
    device = cls_logits.device

    use_ctr = ctr_weight > 0.0
    if use_ctr and box_mode != "ltrb":
        raise ValueError("Centerness is only defined for ltrb box regression.")
    ctr_logits = predictions.get("centerness") if use_ctr else None
    cls_losses, box_losses, ctr_losses = [], [], []

    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)
        if gt_labels.numel() > 0:
            if int(gt_labels.min()) < 0 or int(gt_labels.max()) >= C:
                raise ValueError(
                    f"Target labels must be in [0, {C - 1}], got "
                    f"[{int(gt_labels.min())}, {int(gt_labels.max())}]."
                )

        reg_target, pos_mask, assigned_idx = encode_boxes(
            gt_boxes,
            locations,
            stride,
            center_sampling_radius=(
                min(center_sampling_radius, 1.0)
                if box_mode == "center_size"
                else center_sampling_radius
            ),
            guarantee_gt_match=(box_mode == "center_size"),
        )

        cls_target = torch.zeros(K, C, device=device)
        pos_idx = torch.where(pos_mask)[0]
        if pos_idx.numel() > 0:
            assigned_labels = gt_labels[assigned_idx[pos_idx]]
            cls_target[pos_idx, assigned_labels] = 1.0

        cls_pred = cls_logits[b].permute(1, 2, 0).reshape(K, C)
        box_pred = box_preds[b].permute(1, 2, 0).reshape(K, 4)

        cls_loss = sigmoid_focal_loss(
            cls_pred, cls_target, alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )
        cls_losses.append(cls_loss)

        n_pos = pos_idx.numel()
        if n_pos > 0:
            pred_boxes_decoded = decode_boxes(
                box_pred[pos_idx],
                locations[pos_idx],
                stride,
                box_mode=box_mode,
            )
            gt_boxes_assigned = gt_boxes[assigned_idx[pos_idx]]
            box_loss = giou_loss(pred_boxes_decoded, gt_boxes_assigned)
            box_losses.append(box_loss)

            if use_ctr:
                ctr_pred = ctr_logits[b].permute(1, 2, 0).reshape(K)
                ctr_target = compute_centerness_targets(reg_target, pos_mask)
                ctr_loss = F.binary_cross_entropy_with_logits(
                    ctr_pred[pos_idx], ctr_target[pos_idx], reduction="mean"
                )
                ctr_losses.append(ctr_loss)

    loss_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
    loss_box = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0, device=device)
    loss_ctr_pos = (torch.stack(ctr_losses).mean()
                     if (use_ctr and ctr_losses)
                     else torch.tensor(0.0, device=device))
    loss_ctr = loss_ctr_pos
    total = loss_cls + box_weight * loss_box
    if use_ctr:
        total = total + ctr_weight * loss_ctr
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
    use_centerness: bool = False,
    box_mode: str = "center_size",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_logits = predictions["class_logits"]
    box_preds = predictions["boxes"]

    C = cls_logits.shape[1]
    K = locations.shape[0]

    cls_logits = cls_logits[0].permute(1, 2, 0).reshape(K, C)
    box_preds = box_preds[0].permute(1, 2, 0).reshape(K, 4)

    cls_probs = cls_logits.sigmoid()

    if use_centerness and "centerness" in predictions:
        ctr_logits = predictions["centerness"]
        ctr_preds = ctr_logits[0].permute(1, 2, 0).reshape(K, 1)
        ctr_probs = ctr_preds.sigmoid()
        scores = (cls_probs * ctr_probs).sqrt()  # FCOS: sqrt(cls * ctr)
    else:
        # Paper-aligned: C+4, no centerness — score = class confidence
        scores = cls_probs

    max_scores, max_labels = scores.max(dim=1)

    keep = max_scores > score_threshold
    if not keep.any():
        import sys
        print(f"  [diag] max_cls={cls_probs.max().item():.4f}  "
              f"max_score={max_scores.max().item():.4f}  "
              f"threshold={score_threshold}  → ALL FILTERED",
              file=sys.stderr, flush=True)
        return (
            torch.zeros(0, 4, device=cls_logits.device),
            torch.zeros(0, device=cls_logits.device),
            torch.zeros(0, dtype=torch.long, device=cls_logits.device),
        )

    scores = max_scores[keep]
    labels = max_labels[keep]
    box_preds_kept = box_preds[keep]
    locs = locations[keep]

    boxes = decode_boxes(
        box_preds_kept,
        locs,
        stride,
        max_size=image_size,
        box_mode=box_mode,
    )

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
# Detection metrics
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


def compute_interpolated_ap(recalls: list[float], precisions: list[float]) -> float:
    """COCO-style AP over 101 uniformly spaced recall thresholds."""
    if not recalls:
        return 0.0
    interpolated = []
    for threshold in torch.linspace(0.0, 1.0, 101).tolist():
        candidates = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= threshold
        ]
        interpolated.append(max(candidates, default=0.0))
    return sum(interpolated) / len(interpolated)


def _evaluate_detections_at_iou(
    all_gt: dict[int, list[dict]],
    all_det: dict[int, list[dict]],
    num_classes: int,
    iou_threshold: float,
) -> dict[int, float | None]:
    class_aps: dict[int, float | None] = {}
    for class_id in range(num_classes):
        dets = sorted(
            all_det[class_id],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        gts = all_gt[class_id]
        total_gt = len(gts)
        if total_gt == 0:
            class_aps[class_id] = None
            continue

        gts_by_image: dict[int, list[dict]] = {}
        for gt in gts:
            gts_by_image.setdefault(gt["image_id"], []).append(gt)
        matched = {
            image_id: [False] * len(image_gts)
            for image_id, image_gts in gts_by_image.items()
        }
        true_positives: list[int] = []
        false_positives: list[int] = []
        for det in dets:
            best_iou = 0.0
            best_gt_idx = -1
            image_id = det["image_id"]
            image_gts = gts_by_image.get(image_id, [])
            image_matched = matched.get(image_id, [])
            for gt_idx, gt in enumerate(image_gts):
                if image_matched[gt_idx]:
                    continue
                overlap = compute_iou(det["box"], gt["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                image_matched[best_gt_idx] = True
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        tp_cum = (
            torch.tensor(true_positives).cumsum(dim=0).tolist()
            if true_positives
            else []
        )
        fp_cum = (
            torch.tensor(false_positives).cumsum(dim=0).tolist()
            if false_positives
            else []
        )
        recalls = [tp / total_gt for tp in tp_cum]
        precisions = [
            tp_cum[i] / max(tp_cum[i] + fp_cum[i], 1)
            for i in range(len(tp_cum))
        ]
        class_aps[class_id] = compute_interpolated_ap(recalls, precisions)
    return class_aps


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
    score_threshold: float = 0.001,
    nms_threshold: float = 0.5,
    image_size: int = 512,
    max_samples: Optional[int] = None,
    use_centerness: bool = False,
    box_mode: str = "center_size",
) -> dict[str, float | None]:
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
            if not 0 <= c < num_classes:
                raise ValueError(
                    f"Ground-truth label {c} is outside [0, {num_classes - 1}]."
                )
            all_gt[c].append({"image_id": idx, "box": box})

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
            use_centerness=use_centerness,
            box_mode=box_mode,
        )
        det_boxes, det_scores, det_labels = apply_nms(
            det_boxes, det_scores, det_labels, iou_threshold=nms_threshold,
        )

        for box, score, label in zip(det_boxes, det_scores, det_labels):
            c = int(label.item())
            if not 0 <= c < num_classes:
                raise ValueError(
                    f"Predicted label {c} is outside [0, {num_classes - 1}]."
                )
            all_det[c].append({
                "image_id": idx,
                "confidence": score.item(),
                "box": box.cpu(),
            })

    ap50_by_class = _evaluate_detections_at_iou(
        all_gt,
        all_det,
        num_classes,
        iou_threshold,
    )
    metrics: dict[str, float | None] = {
        f"AP_cls_{class_id}": ap
        for class_id, ap in ap50_by_class.items()
    }
    valid_ap50 = [ap for ap in ap50_by_class.values() if ap is not None]
    metrics["mAP@0.5"] = (
        sum(valid_ap50) / len(valid_ap50)
        if valid_ap50
        else 0.0
    )

    coco_maps = []
    for threshold in [0.50 + 0.05 * idx for idx in range(10)]:
        class_aps = _evaluate_detections_at_iou(
            all_gt,
            all_det,
            num_classes,
            threshold,
        )
        valid = [ap for ap in class_aps.values() if ap is not None]
        coco_maps.append(sum(valid) / len(valid) if valid else 0.0)
    metrics["mAP@[.5:.95]"] = sum(coco_maps) / len(coco_maps)
    return metrics
