# Experiment log - 2026-07-04

## Goal

Raise honest zero-shot RDD Japan-to-Czech mAP@50 toward 50% without using
Czech labels for training or checkpoint selection.

## Environment

- Remote: `root@xj-member.bitahub.com:42462`
- GPU: NVIDIA A100-SXM4 80GB
- Repository: `/root/NEW-PROBE`
- Branch/commit at start: `codex/probe-enhanced-japan-czech`,
  `1b5a66d718efa22f6d2d33cd821414e08aefcc15`

## Changes

1. Replaced the hard-coded `diag_phase3.py` with a CLI that evaluates any
   checkpoint on any configured manifest.
2. Added `scripts/export_yolo_dataset.py` to create a symlinked Ultralytics
   dataset from the existing JSONL manifests. Ultralytics is installed in the
   isolated `/root/yolo-venv`; the PROBE Python environment is unchanged.

## Results

### Existing enhanced checkpoint

Checkpoint:
`checkpoints/enhanced_multilayer_fcos_20e/probe_det_best.pt`

The interrupted run selected this checkpoint at epoch 10 using Japan
source-validation mAP@50 = 19.14%.

Fresh Czech validation evaluation (565 images):

| Metric | Result |
|---|---:|
| mAP@50 | 5.28% |
| mAP@[.5:.95] | 1.27% |
| AP class 0 | 11.69% |
| AP class 1 | 4.70% |
| AP class 2 | 4.10% |
| AP class 3 | 0.63% |

This improves over the completed 6-epoch enhanced result (3.78% mAP@50), but
the source-domain result is still too low to attribute the remaining failure
only to domain shift.

### Dataset diagnostics

- Japan train: 8,400 images; Ultralytics reports 2,060 backgrounds.
- Japan val: 2,098 images; Ultralytics reports 543 backgrounds.
- Czech train: 2,260 images.
- Czech val: 565 images.

### Running control

YOLOv8-s ImageNet/COCO-pretrained source-only control:

- Train: Japan train
- Checkpoint selection: Japan val
- Target evaluation after training: Czech val
- Resolution: 512
- Epochs: 30
- No Czech labels are used for training or selection.

Purpose: separate dataset/protocol limitations from defects in the custom
Phase 3 detector.

Results:

| Evaluation | mAP@50 | mAP@[.5:.95] |
|---|---:|---:|
| Japan validation | 54.52% | 25.30% |
| Czech validation, zero-shot | 13.50% | 4.86% |
| Czech validation, positive images only | 16.90% | 6.03% |

The standard detector is 35.38 points better than the enhanced PROBE detector
on Japan validation, but both retain only about one quarter of source-domain
mAP on Czech. Detection capacity and domain shift are therefore independent
major bottlenecks.

### Czech supervised diagnostic ceiling

YOLOv8-s trained for 30 epochs on Czech train and selected on Czech val reached
approximately 25.6% mAP@50. This target-supervised run is diagnostic only and
is not a zero-shot result. It shows that the paper-reported 88.7% is not
directly comparable with the current manifests/protocol.

### Target-aware pseudo-label experiment

The Japan teacher produced 4,366 candidate boxes at confidence >= 0.05 with
test-time augmentation. Candidate counts were strongly biased:

| Class | Candidates |
|---|---:|
| 0 | 1,003 |
| 1 | 659 |
| 2 | 2,019 |
| 3 | 685 |

To avoid reinforcing the source-biased class-2 distribution, the adaptation
run uses the top 600 predictions per class (2,400 boxes over 1,211 Czech train
images), mixed with all labeled Japan training images. No Czech labels are
read or used. The model is initialized from the source-only YOLO checkpoint
and fine-tuned for 15 epochs with AdamW at `2e-4`; checkpoint selection remains
on Japan validation.

Result: Japan validation remained strong at 55.0% mAP@50, but Czech validation
decreased from 13.50% to 12.70%. Class-balanced confidence filtering does not
make the teacher's incorrect target boxes reliable, so this branch is stopped.

### Partial ViT fine-tuning

The enhanced PROBE configuration was extended with:

- `train_vit_last_n: 4`
- `backbone_lr: 1e-5`

All earlier ViT blocks remain frozen. The last four blocks, final LayerNorm,
prompt projector, and FCOS head are optimized in Phase 3. Ten regression tests
passed before training.

| Variant | Japan mAP@50 | Czech mAP@50 |
|---|---:|---:|
| Frozen ViT, epoch-10 checkpoint | 19.14% | 5.28% |
| Last four ViT blocks trainable | 24.64% | 5.22% |

Partial fine-tuning improves source fitting by 5.5 points but does not improve
target transfer. This isolates domain alignment, rather than source capacity,
as the limiting factor after the detector is strengthened.

### Adaptive BatchNorm

`scripts/adapt_batchnorm.py` updates the source YOLO model's 57 BatchNorm
running-statistic layers using all 2,260 unlabeled Czech train images. No
weights or labels are used.

| Model | Japan mAP@50 | Czech mAP@50 |
|---|---:|---:|
| Source YOLO | 54.52% | 13.50% |
| Source YOLO + Czech AdaBN | 48.10% | 11.80% |

AdaBN harms both domains and is stopped.

## Current conclusions

1. The custom Phase 3 detector is underpowered: a standard YOLO control gains
   about 35 points on Japan validation.
2. Domain shift remains severe even with a strong detector: YOLO retains only
   13.50/54.52 = 24.8% of source mAP on Czech.
3. A 30-epoch Czech-supervised diagnostic reaches only about 25.6% mAP@50.
   The paper's 88.7% result therefore uses an unreleased or materially
   different protocol, split, filtering rule, or recipe.
4. Naive pseudo-labeling and normalization-statistic adaptation reinforce
   target errors. The next implementation must use an EMA teacher with
   weak/strong consistency and an explicit mechanism for class-distribution
   and localization quality, while preserving the strong source detector.

## Completed 100-epoch 640px source baseline

Saved under `checkpoints/yolov8s_japan_100e_640/`.

| Metric | Result |
|---|---:|
| Best Japan mAP@50 (epoch 49) | 53.09% |
| Best Japan mAP@[.5:.95] | 24.13% |
| Final Japan mAP@50 (epoch 100) | 49.92% |
| Czech zero-shot mAP@50 | 11.10% |
| Czech zero-shot mAP@[.5:.95] | 3.97% |

Czech class AP@50:

- longitudinal: 19.8%
- transverse: 12.8%
- alligator: 10.4%
- pothole: 1.4%

The shorter 30-epoch 512px source model remains the stronger cross-domain
teacher (13.50% Czech mAP@50). Longer source training and higher resolution
increase source specialization but do not improve transfer.

## Next experiment gates

1. Train a 100-epoch 640px Czech-supervised oracle for diagnosis only. If its
   mAP@50 remains below 50%, a 50% zero-shot target is not a defensible goal
   under the current manifests and evaluation protocol; resolve the paper's
   split/protocol mismatch before further architecture work.
2. Keep `yolov8s_japan/weights/best.pt` (30 epochs, 512px) as the zero-shot
   teacher baseline because it has the best verified Czech result (13.50%).
3. Implement quality-aware EMA consistency:
   - weak-view EMA teacher and strong-view student;
   - require agreement across two teacher views for class and box location;
   - class-adaptive confidence thresholds;
   - ignore uncertain target regions instead of treating them as background;
   - retain labeled Japan batches to prevent catastrophic forgetting;
   - ramp target consistency weight from zero.
4. Run a 10-epoch smoke experiment. Continue only if Japan mAP@50 stays above
   50% and Czech mAP@50 exceeds 13.50%.
5. Add non-collapsing multi-level feature alignment only after the EMA gate
   passes. Compare fixed random projection, covariance alignment, and
   stop-gradient teacher features one at a time.
6. Three-seed long runs are justified only after a single seed exceeds 20%
   Czech mAP@50. Report 50% as achieved only on the untouched Czech validation
   set without using its labels for training, threshold selection, or
   checkpoint selection.

## Follow-up - 2026-07-05

### Environment and protocol verification

- Remote: `root@xj-member.bitahub.com:42194`
- GPU: NVIDIA A100-SXM4 80GB
- Runtime: PyTorch 2.5.1+cu124, Ultralytics 8.4.87
- Project tests: 11 passed
- The saved Japan 100e/640 checkpoint reproduced exactly on Czech validation:
  11.10% mAP@50 and 3.97% mAP@[.5:.95].

### Czech supervised oracle

This run is a diagnostic upper bound only. Czech labels were used for both
training and validation, so it is not a domain-adaptation result.

- Model: YOLOv8-s, ImageNet/COCO pretrained
- Resolution: 640
- Batch size: 128
- Planned epochs: 100
- Early stopping: epoch 83, with the selected checkpoint from epoch 53

| Metric | Result |
|---|---:|
| Czech mAP@50 | 31.44% |
| Czech mAP@[.5:.95] | 10.83% |
| longitudinal AP50 | 30.0% |
| transverse AP50 | 28.6% |
| alligator AP50 | 42.0% |
| pothole AP50 | 25.3% |

Even fully supervised target training is 18.56 points below the requested 50%
under the current manifests and evaluator. Architecture work remains useful,
but the paper's 88.7% Japan-to-Czech number cannot be treated as directly
comparable until its split, filtering, class mapping, and evaluation protocol
are matched.

The authors' referenced repository was checked again on 2026-07-05:
`https://github.com/xixiaouab/PROBE`, commit `f899bc6`. It contains only the
core scaffold on one branch, with no tags. Its README explicitly states that
the final checkpoints, dataset scripts, and exact benchmark recipes will be
released separately. GitHub therefore does not currently provide the missing
artifacts needed to reproduce the paper's 88.7% result.

### Quality-aware pseudo-label change

`scripts/build_pseudo_dataset.py` now optionally requires agreement between
two teacher prediction views. Predictions must agree on class and box IoU;
matched boxes and confidences are fused before the existing per-class Top-K
selection. Uncertain target images are omitted rather than trained as
background. The mixed dataset retains all labeled Japan images, and the
student's EMA checkpoint becomes the next teacher.
