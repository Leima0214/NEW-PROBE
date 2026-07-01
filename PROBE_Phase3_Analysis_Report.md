# PROBE 项目完整分析报告 — Phase 3 mAP 问题诊断

> **目标**: 供其他模型分析 Phase 3 检测训练中 mAP 无法持续上升的根因。
> **数据集**: United (source, labeled) → Czech (target, unlabeled for SSL, labeled for val)
> **当前状态**: Phase 2 预训练已完成，checkpoint 位于 `checkpoints_from_v100/probe_final.pt`

---

## 1. 项目概述

PROBE (Self-Supervised Visual Prompting for Cross-Domain Road Damage Detection) 是一个跨域道路损害检测方法 (WACV 2026)。

**核心思路**: 从无标签目标域图像中发现视觉原型 (SPEM)，将其注入到冻结的 ViT 中作为 prompt tokens，通过自监督预训练 (SimSiam + Prompt Consistency + DAPA) 适配域差异，最后在少量源域标注数据上训练轻量检测头。

### 三阶段流水线

```
Phase 1: SPEM 原型发现
  unlabeled target images → frozen ViT patch features → PCA → K-means → 10 prototypes (50-dim PCA space)

Phase 2: SSL 预训练 (100 epochs)
  SimSiam (target) + Prompt Consistency (InfoNCE) + DAPA (MMD)
  训练: prompt_projector, ssl_heads, alignment_head
  冻结: ViT backbone

Phase 3: 检测头训练 (50 epochs)  ← 问题阶段
  FCOS-style detection head on source labels
  训练: LightweightDetectionHead only
  冻结: ViT backbone + prompt_projector
```

---

## 2. 项目文件结构

```
PROBE-main/
├── configs/
│   ├── probe_base.yaml              # 基础配置
│   └── probe_japan_czech.yaml       # Japan→Czech 实验配置 (当前使用,已改为 United→Czech)
├── scripts/
│   ├── train.py                     # 训练入口 (三阶段)
│   └── infer.py                     # 推理入口
├── src/probe/
│   ├── models/
│   │   ├── prompts.py               # SPEM (PCA+K-means), PromptProjector, PromptInjector, PromptConsistencyLoss
│   │   └── detector.py              # PromptEnhancedViT, LightweightDetectionHead, PROBEModel
│   ├── engine/
│   │   ├── self_training.py         # Phase 2: SimSiamHeads, DomainAlignmentHead, probe_pretrain_step
│   │   └── detection.py             # Phase 3: FCOS losses, encode/decode boxes, centerness, mAP eval
│   └── data/
│       └── road_damage.py           # RoadDamageDataset (JSONL manifest → PIL → Tensor)
├── data/
│   └── Czech_val_filtered.jsonl     # 验证集 (过滤后, ~200 images)
├── data2/images/                    # 实际图像文件 (Czech + India samples)
└── checkpoints_from_v100/
    └── probe_final.pt               # Phase 2 完成的 checkpoint
```

---

## 3. Phase 1: SPEM 原型发现 (详细)

### 流程
1. 从目标域 (Czech) 无标签图像提取 ViT patch embeddings
2. 对所有 patch tokens 做 PCA 降维 (768 → 50)
3. 在 PCA 空间用 K-means 聚类出 10 个原型中心

### 关键代码 (`prompts.py`)
```python
@dataclass
class PrototypeState:
    mean: torch.Tensor        # [768] PCA mean
    components: torch.Tensor  # [768, 50] PCA 投影矩阵
    centroids: torch.Tensor   # [10, 50] K-means 原型中心 (在 PCA 空间)

class TargetPrototypeDiscovery:
    def fit(self, patch_features):  # patch_features: [N_patches, 768]
        reduced, mean, components = pca_reduce(patch_features, pca_dim=50)
        centroids = kmeans(reduced, num_prototypes=10)
        return PrototypeState(mean, components, centroids)

class PromptProjector(nn.Module):
    """2-layer MLP: PCA prototypes (50-dim) → ViT prompt tokens (768-dim)"""
    def __init__(self):
        self.net = Sequential(
            Linear(50, 256), GELU(), Linear(256, 768)
        )
```

### 配置参数
```yaml
spem:
  pca_dim: 50
  num_prototypes: 10
  kmeans_iters: 25
  prompt_hidden_dim: 256
  injection_layers: [0, 6]    # 在 ViT 第 0 和第 6 层注入 prompts
  prompt_temperature: 0.2
  prompt_weight: 1.0
```

### Checkpoint 中 prototypes 的状态
```
centroids shape: [10, 50]
centroids: 无 NaN, 无 Inf, 值范围合理 (约 [-10, +10])
mean shape: [768], components shape: [768, 50]
结论: prototypes 正常
```

---

## 4. Phase 2: SSL 预训练 (详细)

### 三个损失函数

**A. SimSiam Loss** (target domain only)
```
对同一张 target 图像的两次增强 (view1, view2)
  z1 = projector(view1_features), p1 = predictor(z1)
  z2 = projector(view2_features), p2 = predictor(z2)
  loss = 0.5 * [neg_cos_sim(p1, z2.detach()) + neg_cos_sim(p2, z1.detach())]
```

**B. Prompt Consistency Loss** (InfoNCE)
```
image_features = CLS token (prompt-injected)
prompt_means = mean of processed prompt tokens from each image
logits = normalized(image_features) @ normalized(prompt_means).T / temperature
labels = [0, 1, ..., B-1]  (每个 image 和自己的 prompt 是正对)
loss = CrossEntropy(logits, labels)
```

**C. DAPA Loss** (Domain Alignment via Prompt-conditioned Alignment)
```
source_projected = alignment_head(source_features)
target_projected = alignment_head(target_features)
loss = ||mean(source_projected) - mean(target_projected)||^2  (linear-kernel MMD)
```

### 总损失
```
total = λ_ssl * 1.0 + λ_prompt * 1.0 + λ_dapa * 0.5
```

### 训练设置
```yaml
optim:
  lr: 0.0005
  weight_decay: 0.05
  pretrain_epochs: 100
data:
  batch_size: 8
  num_workers: 4
```

### 可训练参数
- `prompt_projector` (2-layer MLP)
- `ssl_heads` (projector + predictor)
- `alignment_head` (2-layer MLP)
- ViT backbone: **冻结**

---

## 5. Phase 3: 检测头训练 (重点分析)

### 5.1 整体流程

```
1. 加载 Phase 2 checkpoint → backbone (ViT + prompt_projector) 权重 + prototype_state
2. 冻结 backbone 所有参数
3. 初始化 LightweightDetectionHead (随机)
4. 在 source (United) 标注数据上训练, target (Czech) 验证集上评估 mAP
```

### 5.2 LightweightDetectionHead 架构

```python
class LightweightDetectionHead(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=384, neck_dim=128, num_classes=5, cls_prior=0.1):
        # 共享 Stem
        self.stem = Sequential(
            Conv2d(768, 384, 3, padding=1, bias=False),
            BatchNorm2d(384),
            GELU()
        )
        # 三个独立分支 (结构相同)
        self.cls_branch = Sequential(       # 输出: [B, 5, 32, 32]
            Conv2d(384, 128, 1), GELU(), Conv2d(128, 5, 1)
        )
        self.box_branch = Sequential(       # 输出: [B, 4, 32, 32]
            Conv2d(384, 128, 1), GELU(), Conv2d(128, 4, 1)
        )
        self.ctr_branch = Sequential(       # 输出: [B, 1, 32, 32]
            Conv2d(384, 128, 1), GELU(), Conv2d(128, 1, 1)
        )
        # 初始化: cls_branch 最后一层 bias = ln(cls_prior/(1-cls_prior))
        #          box/ctr 最后一层 bias = 0.0
        #          所有权重: N(0, 0.01)

    def forward(self, patch_tokens):  # patch_tokens: [B, 1024, 768]
        side = 32  # sqrt(1024)
        feature_map = patch_tokens.transpose(1,2).reshape(B, 768, 32, 32)
        shared = self.stem(feature_map)    # [B, 384, 32, 32]
        return {
            "class_logits": self.cls_branch(shared),   # [B, 5, 32, 32]
            "boxes": self.box_branch(shared),          # [B, 4, 32, 32]
            "centerness": self.ctr_branch(shared),     # [B, 1, 32, 32]
        }
```

### 5.3 输入特征来源

```
Image [B, 3, 512, 512]
  → ViT patch_embed: [B, 1024, 768]
  → + CLS token + pos_embed: [B, 1025, 768]
  → Prompt Injector 在 layers [0, 6] 注入 10 个 prompt tokens
    序列: [CLS, prompt_0..9, patch_0..1023] → attention → [CLS', prompt'_0..9, patch'_0..1023]
  → 移除 prompt tokens: [CLS', patch'_0..1023]
  → ViT LayerNorm
  → patch_tokens = tokens[:, 1:, :]  # [B, 1024, 768] — 只取 patch tokens, 丢弃 CLS
  → DetectionHead(patch_tokens)
```

**关键**: prompt tokens 通过与 patch tokens 的 self-attention 交互来影响 patch features。
prompt_projector 在 Phase 2 中训练，Phase 3 中保持冻结。

### 5.4 FCOS 损失函数 (detection_loss)

```python
def detection_loss(predictions, targets, locations, stride, ...):
    """
    predictions:
      class_logits: [B, 5, 32, 32] — 每个位置对 5 个类的二分类 logits
      boxes:        [B, 4, 32, 32] — [l, t, r, b] 距离的 log 值 (单位: stride)
      centerness:   [B, 1, 32, 32] — centerness logits

    targets: 每张图 [{"boxes": [N, 4], "labels": [N]}]

    locations: [1024, 2] — 每个 grid cell 中心的 (x, y) 像素坐标
    stride: 16.0 (512/32)
    """
    for each image in batch:
        # 1. Box 编码: 将 GT boxes 分配到 grid locations
        reg_target, pos_mask, assigned_idx = encode_boxes(gt_boxes, locations, stride)
        # reg_target: [1024, 4] — [l, t, r, b]/stride
        # pos_mask: [1024] bool
        # assigned_idx: [1024] — 每个位置被分配到的 GT box 索引

        # 2. 分类损失 (Focal Loss)
        cls_target = one_hot_enoded_assigned_class  # [1024, 5]
        cls_loss = sigmoid_focal_loss(cls_pred, cls_target, α=0.25, γ=2.0)
        # reduction="mean" → 除以正样本数

        # 3. Box 回归损失 (GIoU, 仅正样本)
        pred_boxes_decoded = exp(box_pred[pos]) * stride → pixel coords
        box_loss = GIoU(pred_boxes_decoded, gt_boxes_assigned)

        # 4. Centerness 损失 (BCE, 仅正样本)
        ctr_target = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))  # [1024]
        ctr_loss = BCE(ctr_pred[pos], ctr_target[pos])

        # 5. Centerness 负样本弱监督 (新增)
        ctr_loss_neg = 0.1 * BCE(ctr_pred[neg], zeros)

    total = cls_loss + 1.0 * box_loss + 1.0 * (ctr_loss + 0.1 * ctr_loss_neg)
```

### 5.5 Box 编码/解码 (FCOS 标准)

**编码 (encode_boxes)**:
```
对每个位置 i 和每个 GT box j:
  l = x_ctr[i] - x1[j]   (grid中心到box左边的距离，像素)
  t = y_ctr[i] - y1[j]
  r = x2[j] - x_ctr[i]
  b = y2[j] - y_ctr[i]

inside[i,j] = (l >= 0) & (t >= 0) & (r >= 0) & (b >= 0)  # >= (已修复,原为 >)

分配策略: 对位置 i, 在所有 inside[i,j]=True 的 box 中选择面积最小的
  assigned_idx[i] = argmin_j(area[j] if inside[i,j] else ∞)

targets[i] = [l, t, r, b] / stride   (stride=16.0, 单位归一化)
```

**解码 (decode_boxes)**:
```
[l, t, r, b] = exp(box_preds) * stride    (log空间 → 像素距离)
x1 = x_ctr - l,  y1 = y_ctr - t
x2 = x_ctr + r,  y2 = y_ctr + b
clamp to [0, 512]
```

### 5.6 Centerness 目标计算

```python
def compute_centerness_targets(targets, mask):
    """
    targets: [K, 4] — [l, t, r, b]/stride
    返回: [K] — centerness ∈ [0, 1]

    公式: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
    """
    lr = [targets[:,0], targets[:,2]]  # [l, r]
    tb = [targets[:,1], targets[:,3]]  # [t, b]
    centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
    centerness[~mask] = 0.0   # 负样本位置 = 0
    return centerness
```

**诊断结果**: 正样本 centerness 分布: min=0.098, max=0.786, mean=0.361
大多数 centerness 目标在 [0.2, 0.5] 范围内 (stride=16 导致很多 grid 位置靠近 box 边缘)

### 5.7 推理时的评分 (collect_detections)

```python
def collect_detections(predictions, locations, stride, score_threshold=0.05, max_detections=196):
    cls_logits: [1, 5, 32, 32] → reshape → [1024, 5]
    box_preds:  [1, 4, 32, 32] → reshape → [1024, 4]
    ctr_logits: [1, 1, 32, 32] → reshape → [1024, 1]

    # 评分
    cls_probs = sigmoid(cls_logits)           # [1024, 5]
    ctr_probs = sigmoid(ctr_logits)           # [1024, 1]
    scores = sqrt(cls_probs * ctr_probs)      # [1024, 5] ← 几何平均
    max_scores, max_labels = scores.max(dim=1) # [1024], [1024]

    # 筛选
    keep = max_scores > 0.05                  # 分数阈值
    boxes = decode_boxes(box_preds[keep], locations[keep], stride)
    # 保留 top-196, 过滤非法框 (x2>x1, y2>y1)

    return top_196_boxes, top_196_scores, top_196_labels
```

### 5.8 mAP 评估 (evaluate_map)

```
1. 对 val 数据集中每张图:
   - model.detect(image) → class_logits, boxes, centerness
   - collect_detections(predictions, score_threshold=0.05, max_detections=196)
   - apply_nms(boxes, scores, labels, iou_threshold=0.5, max_detections=100)
   - 记录检测结果 (box, score, label)

2. 对每个类别 c (0-4):
   - 按 confidence 降序排列所有检测
   - 对每个检测, 找同图同类中 IoU ≥ 0.5 且未被匹配的 GT box
   - 匹配成功 → TP, 失败 → FP
   - 计算 PR 曲线

3. VOC 11-point interpolated AP:
   - 对 11 个 recall 阈值 {0, 0.1, ..., 1.0}, 取 max precision where recall ≥ t
   - AP = mean of 11 interpolated precisions

4. mAP@0.5 = mean(AP_c for c in [0..4]) / 5
```

---

## 6. Phase 3 训练循环 (train_detection_head)

### 完整训练循环

```python
def train_detection_head(model, prototype_state, cfg, args, device):
    # --- 初始化 ---
    image_size = 512
    feature_size = 32        # 512 // 16
    stride = 16.0            # 512 / 32
    locations = generate_grid(32, 16.0, device)  # [1024, 2], 每个 cell 中心像素坐标

    # --- 数据集 ---
    source_dataset = RoadDamageDataset(source_manifest, image_root,
                                        transform=eval_transform(512), image_size=512)
    val_dataset = RoadDamageDataset(val_manifest, image_root,
                                     transform=eval_transform(512), image_size=512)

    # --- 优化器 (仅检测头) ---
    optimizer = AdamW(model.detection_head.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # --- 训练参数 ---
    total_epochs = 50
    focal_alpha = 0.25, focal_gamma = 2.0
    box_weight = 1.0, ctr_weight = 1.0
    score_threshold = 0.05, nms_threshold = 0.5
    val_interval = 5
    grad_accum = args.grad_accum  (default 1)

    # 冻结 backbone
    model.backbone.freeze_backbone()
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(50):
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, targets) in enumerate(source_loader):
            images = images.to(device)            # [B, 3, 512, 512]

            # 前向传播
            _, patch_tokens, _ = model.encode(images, prototype_state)
            predictions = model.detection_head(patch_tokens)

            # 损失 + 反向传播
            loss_dict = detection_loss(predictions, targets, locations, stride, ...)
            (loss_dict["det_total"] / grad_accum).backward()

            # 梯度累积
            if (batch_idx + 1) % grad_accum == 0:
                clip_grad_norm_(detection_head.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # 每 5 epochs 验证
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_map(model, val_dataset, prototype_state, ...)
            mAP = metrics["mAP@0.5"]
```

### 数据流示意图

```
Source Image (United, 512x512)
    ↓ eval_transform (Resize + Normalize)
    ↓
    ↓ model.encode(image, prototype_state)
    ↓    ├─ ViT patchify: [B, 1024, 768]
    ↓    ├─ PromptProjector(centroids) → 10 prompt tokens [B, 10, 768]
    ↓    ├─ Inject prompts at layers 0, 6
    ↓    └─ ViT blocks → LayerNorm → patch_tokens [B, 1024, 768]
    ↓
    ↓ model.detection_head(patch_tokens)
    ↓    ├─ reshape → [B, 768, 32, 32]
    ↓    ├─ stem: Conv-BN-GELU → [B, 384, 32, 32]
    ↓    ├─ cls_branch → [B, 5, 32, 32]
    ↓    ├─ box_branch → [B, 4, 32, 32]
    ↓    └─ ctr_branch → [B, 1, 32, 32]
    ↓
    ↓ detection_loss(predictions, GT boxes/labels)
    ↓    ├─ encode_boxes → assign locations to GT boxes
    ↓    ├─ Focal Loss (classification)
    ↓    ├─ GIoU Loss (box regression)
    ↓    ├─ BCE Loss (centerness, positive positions)
    ↓    └─ BCE Loss (centerness, negative positions, weight=0.1)
    ↓
    ↓ backward() → optimizer.step()
```

---

## 7. 关键数值诊断

### 7.1 初始化时各分支输出

| 分支 | 初始化 bias | sigmoid 后 | 说明 |
|------|-----------|-----------|------|
| cls_branch | `ln(0.1/0.9) = -2.197` | **0.100** | cls_prior=0.1 (已从 0.01 改为 0.1) |
| ctr_branch | `0.0` | **0.500** | 所有位置初始 centerness=0.5 |
| box_branch | `0.0` | N/A (log space) | 初始预测 [l,t,r,b] 在 log 空间 ≈ 0 |

### 7.2 初始化时的评分分布

| 配置 | cls_prior | 评分公式 | 分数范围 | 通过 0.05 阈值 | 检测数 |
|------|-----------|---------|---------|--------------|--------|
| 原始 (bug) | 0.01 | sqrt | [0.069, 0.084] | 1024/1024 | 196 |
| 旧方案 | 0.01 | linear | [0.005, 0.006] | **0/1024** | **0** |
| **当前** | **0.1** | **sqrt** | **[0.220, 0.241]** | 1024/1024 | 196 |

### 7.3 损失在初始化时的分解 (一个典型 batch)

| 损失项 | 值 | 占总损失比例 |
|--------|-----|------------|
| det_cls (Focal) | ~1.13 | ~40% |
| det_box (GIoU) | ~0.94 | ~33% |
| det_ctr (centerness pos) | ~0.69 | ~25% |
| det_ctr_neg (centerness neg, ×0.1) | ~0.07 | ~2% |
| **Total** | **~2.83** | **100%** |

### 7.4 Centerness 目标的分布 (针对典型 80×80px box)

```
centerness ∈ [0, 1], 目标是 sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))

统计 (32×32 grid, stride=16):
  < 0.2:  13/65 (20%)  — 边缘位置
  0.2-0.5: 39/65 (60%) — 中间位置
  0.5-0.8: 13/65 (20%) — 近中心位置
  ≥ 0.8:   0/65  (0%)  — 几乎不存在 (stride=16 太粗糙)

结论: 大部分正样本位置的 centerness 目标在 0.2-0.5 之间
      而负样本位置无监督 → centerness 保持在 0.5
      → 负样本的 centerness(0.5) 平均上高于正样本目标(0.36)
      → 这就是为什么需要负样本 centerness 监督
```

---

## 8. 已知问题与修复状态

### 8.1 已修复的问题

| # | 问题 | 文件 | 修复 | 状态 |
|---|------|------|------|------|
| 1 | `encode_boxes` 使用 `>` 排除边界位置 | detection.py:61 | 改为 `>=` | ✅ |
| 2 | `ctr_weight=3.0` 使 centerness 损失主导 | probe_japan_czech.yaml:40 | 改为 `1.0` | ✅ |
| 3 | 负样本 centerness 无监督 (噪声底板 0.5) | detection.py:205-211 | 添加 0.1×BCE 负监督 | ✅ |
| 4 | cls_prior=0.01 过于悲观 | probe_japan_czech.yaml:31 | 改为 `0.1` (bias=-2.197) | ✅ |
| 5 | Phase 3 无 gradient accumulation | train.py:406-450 | 添加 grad_accum 支持 | ✅ |
| 6 | 评分未用 sqrt (与 cls_prior 配合) | detection.py:251 | 保持 `sqrt` (配合 cls_prior=0.1) | ✅ |

### 8.2 仍需关注的问题

#### A. Centerness 正样本目标偏低
由于 stride=16 (粗粒度), 32×32 grid 中多数正样本位置位于 box 边缘附近。
centerness 目标均值仅 0.361, 而负样本 (经弱监督后) 接近 0。
这可能导致模型对 "centerness 应该有多高" 信号不够强。

#### B. 单一检测尺度
FCOS 通常使用 FPN 多尺度特征, 但当前实现只有一个 32×32 特征图。
这意味着:
- 小物体 (边长 < 16px) 可能完全没有正样本位置
- 大物体的边界定位精度有限 (stride=16 是最小分辨率)

#### C. 验证集大小
`Czech_val_filtered.jsonl` 仅 ~200 张图。如果类别分布不均, 某些类的 AP 会很不可靠。

#### D. Prompt 注入的有效性
prompt_projector 在 Phase 2 中训练, 但其输出 (prompt tokens) 对 source 域 (United) 图像的有效性未经验证。
如果 United 和 Czech 的道路场景差异很大, 从 Czech 图像学到的 prototypes 可能对 United 图像没有帮助。

#### E. BatchNorm 在冻结 backbone 场景下
DetectionHead 中的 BatchNorm 在训练时用 batch stats, 评估时用 running stats。
由于 backbone 冻结, 输入特征分布是固定的。但 BatchNorm 仍然在适应中, 前几个 epoch 的 train/eval 模式差异可能影响 mAP。

---

## 9. Phase 3 完整配置参数

```yaml
# 检测头架构
detection:
  num_classes: 5           # 道路损害类别数 (0-4)
  hidden_dim: 384          # Stem 输出通道
  neck_dim: 128            # 分支中间通道
  cls_prior: 0.1           # 分类先验概率 (bias init = -2.197)

# 检测训练
detection_optim:
  lr: 0.0003               # AdamW 学习率
  lr_min: 0.000001         # CosineAnnealing 最小 lr
  weight_decay: 0.0001
  epochs: 50
  focal_alpha: 0.25        # Focal Loss alpha
  focal_gamma: 2.0         # Focal Loss gamma
  box_weight: 1.0          # GIoU loss 权重
  ctr_weight: 1.0          # Centerness loss 权重 (已从 3.0 降低)
  score_threshold: 0.05    # 推理分数阈值
  nms_threshold: 0.5       # NMS IoU 阈值
  val_interval: 5          # 每 N epochs 验证一次

# 数据
data:
  source_manifest: data/Japan_train.jsonl    # 实际使用 United 数据
  target_manifest: data/Czech_unlabeled.jsonl
  val_manifest: data/Czech_val_filtered.jsonl
  image_root: images
  batch_size: 8
  num_workers: 4
```

---

## 10. 关键代码路径速查

| 功能 | 文件:行号 |
|------|----------|
| 训练入口 Phase 3 | `scripts/train.py:325-523` |
| 模型构建 + cls_prior 读取 | `scripts/train.py:594-600` |
| DetectionHead 定义 | `src/probe/models/detector.py:82-151` |
| encode_boxes (≥ 边界) | `src/probe/engine/detection.py:39-78` |
| compute_centerness_targets | `src/probe/engine/detection.py:100-112` |
| Focal Loss (mean by pos) | `src/probe/engine/detection.py:119-140` |
| detection_loss (完整) | `src/probe/engine/detection.py:151-214` |
| collect_detections (sqrt 评分) | `src/probe/engine/detection.py:217-265` |
| apply_nms | `src/probe/engine/detection.py:268-280` |
| evaluate_map (VOC 11-point) | `src/probe/engine/detection.py:314-413` |
| PROBEModel.encode | `src/probe/models/detector.py:166-171` |
| PromptEnhancedViT.forward_tokens | `src/probe/models/detector.py:47-69` |
| RoadDamageDataset (box scaling) | `src/probe/data/road_damage.py:12-58` |
| Phase 2 checkpoint 加载 | `scripts/train.py:606-636` |

---

## 11. mAP 无法持续上升的可能原因 (排查方向)

### 可能性排序

1. **Backbone 特征不够判别性** (最可能)
   - ViT 全程冻结 (只是 ImageNet 预训练), 从未见过道路损害图像
   - Prompt tokens 从 Czech 图像学到, 注入到 United 图像可能不匹配
   - 验证: 检查 Phase 2 训练损失是否有效下降

2. **单一尺度检测限制** (很可能)
   - 32×32 grid + stride=16 对小物体不友好
   - 道路损害 (裂缝、坑洞) 可能小于 16px
   - 验证: 检查验证集中 GT box 的尺寸分布

3. **Centerness 与分类解耦不充分** (可能)
   - 推理时 centerness 乘以分类分数, 但训练时两者独立优化
   - 如果一个好了另一个没跟上, 最终分数不会提高
   - 验证: 分别检查各 epoch 的 cls_loss / box_loss / ctr_loss 趋势

4. **类别不平衡** (可能)
   - 5 个类的样本数可能严重不平衡
   - Focal Loss 的 `alpha=0.25` 可能不适应当前分布
   - 验证: 检查 source 和 val 集合的类别分布

5. **Score=0.05 阈值仍偏低** (可能)
   - 1024 个位置中 196 个被保留, 引入了大量低质量检测
   - 验证: 尝试提高 score_threshold 到 0.1 或 0.2

6. **Validation 域偏移** (可能)
   - Source = United, Val = Czech (不同国家)
   - Prompt 从 Czech 学到, 但检测头在 United 数据上训练
   - 检测头可能在 source 域上表现好, 但在 target 域上泛化差
   - 验证: 在 source 验证集上评估 mAP

---

*报告生成时间: 2026-06-30*
*项目 commit: e31bbb2 "add centerness branch + filtered val + ctr_weight=3.0"*
