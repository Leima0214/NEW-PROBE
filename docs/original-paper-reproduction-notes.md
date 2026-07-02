# PROBE 原始文献复现笔记

原始文献：Xi Xiao et al., "Self-Supervised Visual Prompting for Cross-Domain
Road Damage Detection", WACV 2026, pp. 3514-3524。

本笔记将信息分为三类：

- **论文明确**：可以作为复现的硬约束。
- **论文未披露**：不能把当前代码或常见做法误称为论文设置。
- **本地实现现状**：当前项目中的工程补全方案，后续需要实验验证。

## 1. 任务与整体流程

PROBE 面向封闭类别集、单源域到单目标域的无监督域适应道路损伤检测。

1. Phase 1 / SPEM 原型发现：用冻结 ViT 从无标签目标域图像提取 patch
   embeddings，先将维度从 `D=768` 经 PCA 降至 `d'=50`，再做 K-means，
   得到 `K=10` 个目标域视觉原型。
2. Phase 2 / 自监督适配：两层 MLP + GELU 将原型投影回 ViT embedding
   空间，在浅层 `L0` 和中层 `L6` 注入 prompt。冻结 ViT，只训练 prompt
   projector、SimSiam projector/predictor 和 DAPA projection head。
3. Phase 3 / 检测头训练：冻结 prompt-enhanced ViT，只用有标签源域样本训练
   轻量检测头。零样本设置直接在目标域测试；few-shot 设置只微调检测头。

Phase 2 总损失：

`L_total = L_ssl + lambda_1 * L_prompt + lambda_2 * L_DAPA`

论文选定 `lambda_1=1.0`、`lambda_2=0.5`。DAPA 是 prompt-conditioned
source/target features 经小型投影头后的线性核 MMD，即两个 batch 均值的平方
欧氏距离。ViT backbone 在全部预训练过程中冻结。

## 2. Phase 3 的论文硬约束

### 2.1 输入特征

- 输入是 prompt-enhanced ViT 最后一层的 **patch tokens** `z^(L)`，不是
  `[CLS]` token。
- ViT-B/16 的 embedding dimension 是 `768`。
- patch tokens 被还原为二维 feature map。
- 论文用 `224x224 -> 14x14x768` 举例；主实验统一使用 `512x512`，因此按
  patch size 16 推导，实际应为 `32x32x768`。
- Phase 3 冻结 prompt-enhanced ViT，即 ViT 和 Phase 2 训练得到的 prompt
  模块都不再更新，只训练 detection head。

### 2.2 检测头结构

论文 Section 3.5 和 Figure 1 明确给出单路三阶段结构：

```text
patch feature map
  -> Conv-BN-GELU, 3x3, 768 -> 384
  -> Conv-GELU,    1x1, 384 -> 128
  -> Prediction,   1x1, 128 -> C+4
```

- `C` 是缺陷类别数。
- 额外 4 个通道参数化 bounding box。
- 论文没有 objectness 或 centerness 输出，因此严格对齐输出必须是 `C+4`。

### 2.3 损失与训练数据

- 分类损失：Focal Loss。
- 框回归损失：GIoU Loss。
- 训练数据：有标签源域子集。
- 目标域零样本评估：Phase 3 不使用目标域标签。
- Few-shot：保持 backbone 冻结，只用目标域 `1% / 5% / 10%` 标签微调
  detection head；论文也在实验设置段单独提到 5%。

## 3. Phase 3 中论文没有披露的信息

以下内容均不能从论文正文唯一确定：

- Focal Loss 的 `alpha`、`gamma` 和分类/回归损失权重。
- bounding box 的 4 个参数究竟是 `xyxy`、`cxcywh`、grid-relative offsets
  还是 `ltrb` distances，以及激活函数和尺度归一化。
- 每个网格位置如何匹配 GT box；没有 anchor、正负样本分配或 center
  sampling 说明。
- 是否有背景类、分类输出采用 softmax 还是独立 sigmoid。
- 推理阈值、NMS 类型及 IoU 阈值、每图最大检测数。
- 检测头优化器、学习率、weight decay、scheduler、warmup、batch size、
  epoch 数、梯度裁剪和训练增强。
- 主实验所谓 "small labeled subset of source" 的具体比例。
- 预训练权重来源和全部随机种子细节。
- COCO-style `mAP@[.5:.95]` 的完整结果和实现细节。主表实际只列一个
  `mAP` 和 precision，正文对指标口径的描述不足以消除歧义。

因此，任何能跑通的 Phase 3 都必须做工程假设，并通过源域过拟合、消融和目标域
评估验证，不能声称是官方精确实现。

## 4. 当前本地 Phase 3 与论文的差异

当前代码已经不是 README 所称的纯 scaffold，而是补入了一套 FCOS 风格方案：

- `LightweightDetectionHead` 使用分类/回归双塔；每塔默认 3 个 `3x3 Conv +
  GroupNorm + GELU`，hidden dim 256，输出层也是 `3x3`。
- 论文则是共享单路 `3x3 Conv-BN-GELU (384) -> 1x1 Conv-GELU (128) ->
  1x1 (C+4)`。
- 当前 box 编码使用单尺度 grid 上的 `ltrb` distances、最小面积 GT 分配、
  圆形 center sampling 和 softplus decode；这些都是合理工程选择，但论文未写。
- 当前实现支持可选 centerness；严格论文对齐必须设 `ctr_weight=0`。
- 当前验证是 VOC 2007 11-point `mAP@0.5`，并非论文宣称同时报告的 COCO
  `mAP@[.5:.95]`。
- `source_label_fraction` 虽写在配置中，但 Phase 3 数据加载当前没有使用该
  参数，实际上会使用 manifest 中全部源域样本。
- `probe_a100.yaml` 中把 `lr=1e-3`、`epochs=100` 注释成 paper-aligned，
  但论文没有披露这两个值；它们属于本地假设。
- `probe_a100.yaml` 的 `num_classes=4`，`probe_base.yaml` 和
  `probe_japan_czech.yaml` 为 5。类别数必须由具体 source-target 闭集映射
  决定，不能混用。
- `scripts/infer.py` 仍按旧接口传入 `neck_dim`，与当前检测头构造函数不一致，
  后续 Phase 3 联调时需要修复。

## 5. 论文实验锚点

主表统一条件：`512x512`、NVIDIA A100、FP16；结果取 3 个随机种子平均。

PROBE 报告的目标域结果：

| Target | mAP | Precision |
|---|---:|---:|
| TD-RD | 90.2 | 90.5 |
| CNRDD | 38.1 | 47.2 |
| CRDDC'22 | 50.3 | 55.1 |

消融锚点：

| Variant | CNRDD mAP | CRDDC'22 mAP |
|---|---:|---:|
| Source-only | 23.1 | 42.5 |
| + SSL | 27.5 | 46.2 |
| + SSL + SPEM | 33.8 | 48.1 |
| + SSL + DAPA | 34.5 | 48.7 |
| + SSL + SPEM + DAPA | 38.1 | 50.3 |

Prompt 设计锚点：

- `K=1/5/10/15` 在 CNRDD 上为 `31.5/36.8/38.1/37.7`。
- `K=10` 最优，主实验采用该值。
- `L0`、`L6`、`L0+L6` 在 CNRDD 上为 `36.5/37.1/38.1`，主实验采用
  Shallow+Mid。

CRDDC'22 few-shot mAP：

| Labels | 0% | 1% | 5% | 10% |
|---|---:|---:|---:|---:|
| PROBE | 50.3 | 54.5 | 57.0 | 58.5 |

## 6. 后续 Phase 3 实现原则

1. 先实现一条严格论文结构基线：单路 `768->384->128->C+4`、BN、Focal +
   GIoU、无 centerness。
2. 将论文未披露项全部配置化，并明确标注为 reproduction assumptions。
3. 首先在少量源域训练图像上做过拟合测试；若源域训练集 mAP 仍低，禁止把问题
   归因于跨域表示。
4. 使用标准 COCO evaluator 同时报 `mAP@50` 和 `mAP@[.5:.95]`，另保留
   class-wise precision/recall；不要用 VOC 11-point AP 冒充论文指标。
5. 固定类别映射、图像缩放后的 box 同步变换、源域采样比例和三组随机种子。
6. 论文严格基线跑通后，再把 FCOS 双塔、center sampling、centerness 等作为
   明确的工程增强逐项消融，比较其是否更接近论文数值。

## 7. 复现风险判断

论文对 Phase 3 的描述不足以直接重建唯一检测器。达到论文数值的主要风险不只是
检测头代码，还包括未公开的数据划分、类别映射、源域标签比例、训练 recipe 和
指标实现。后续目标应分两层：

- **方法复现**：严格满足论文公开结构与损失，训练和推理可运行、结果可验证。
- **数值复现**：围绕未披露项做受控搜索，以论文消融和主表作为锚点逐步逼近。
