from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .prompts import PromptInjector, PromptProjector, PrototypeState


@dataclass
class DetectionBatch:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor | None = None


class PromptEnhancedViT(nn.Module):
    """Frozen ViT wrapper with SPEM prompt injection."""

    def __init__(
        self,
        vit: nn.Module,
        prompt_projector: PromptProjector,
        injection_layers: tuple[int, ...] = (0, 6),
        detection_layers: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.vit = vit
        self.prompt_projector = prompt_projector
        self.injector = PromptInjector(injection_layers)
        self.detection_layers = set(detection_layers)
        if any(
            layer < 0 or layer >= len(self.vit.blocks)
            for layer in self.detection_layers
        ):
            raise ValueError("detection_layers contains an invalid ViT block index.")
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.vit.eval()

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vit.patch_embed(images)
        if patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        cls = self.vit.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        return tokens + self.vit.pos_embed[:, : tokens.shape[1]]

    def forward_tokens(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patchify(images)
        prompts = self.prompt_projector(
            prototype_state.centroids.to(tokens.device),
            batch_size=images.shape[0],
        )
        processed_prompts = prompts
        detection_tokens = []

        for layer_id, block in enumerate(self.vit.blocks):
            if self.injector.should_inject(layer_id):
                K = prompts.shape[1]
                tokens = self.injector.insert(tokens, prompts)
                tokens = block(tokens)
                processed_prompts = tokens[:, :K, :]
                tokens = self.injector.remove(tokens, K)
            else:
                tokens = block(tokens)
            if layer_id in self.detection_layers:
                detection_tokens.append(self.vit.norm(tokens)[:, 1:])

        tokens = self.vit.norm(tokens)
        patch_tokens = (
            torch.stack(detection_tokens).mean(dim=0)
            if detection_tokens
            else tokens[:, 1:]
        )
        return tokens, patch_tokens, processed_prompts

    def forward(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, patch_tokens, processed_prompts = self.forward_tokens(
            images,
            prototype_state,
        )
        image_features = tokens[:, 0]
        return image_features, patch_tokens, processed_prompts


class LightweightDetectionHead(nn.Module):
    """Lightweight dense detector over the final ViT patch feature map.

    ``architecture="paper"`` implements Section 3.5 exactly at the module
    level: 3x3 Conv-BN-GELU (768->384), 1x1 Conv-GELU (384->128), then a
    1x1 C+4 prediction layer. ``architecture="fcos"`` retains the earlier
    two-tower implementation for controlled ablations and old experiments.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 5,
        cls_prior: float = 0.01,
        head_depth: int = 3,
        num_groups: int = 8,
        use_centerness: bool = False,
        architecture: str = "paper",
        paper_mid_dim: int = 384,
        paper_neck_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_centerness = use_centerness
        self.architecture = architecture

        if architecture not in {"paper", "fcos"}:
            raise ValueError(f"Unknown detection head architecture: {architecture}")
        if architecture == "paper" and use_centerness:
            raise ValueError("The paper C+4 detection head does not use centerness.")

        if architecture == "paper":
            self.shared_head = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    paper_mid_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(paper_mid_dim),
                nn.GELU(),
                nn.Conv2d(paper_mid_dim, paper_neck_dim, kernel_size=1),
                nn.GELU(),
            )
            self.prediction = nn.Conv2d(
                paper_neck_dim,
                num_classes + 4,
                kernel_size=1,
            )
            self.cls_tower = None
            self.reg_tower = None
            self.cls_logits = None
            self.box_logits = None
            self.ctr_logits = None
            self._init_weights(cls_prior)
            return

        # --- Classification tower -------------------------------------------
        cls_layers: list[nn.Module] = []
        in_dim = embed_dim
        for i in range(head_depth):
            out_dim = hidden_dim
            cls_layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(num_groups, out_dim), out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1)

        # --- Regression tower (shared by box + centerness) -----------------
        reg_layers: list[nn.Module] = []
        in_dim = embed_dim
        for i in range(head_depth):
            out_dim = hidden_dim
            reg_layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(num_groups, out_dim), out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        self.reg_tower = nn.Sequential(*reg_layers)
        self.box_logits = nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1)
        if use_centerness:
            self.ctr_logits = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        else:
            self.ctr_logits = None  # paper-aligned: C+4, no centerness

        self._init_weights(cls_prior)

    def _init_weights(self, cls_prior: float = 0.01) -> None:
        if self.architecture == "paper":
            for module in self.shared_head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            nn.init.normal_(self.prediction.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.prediction.bias, 0.0)
            bias_value = math.log(cls_prior / (1.0 - cls_prior))
            with torch.no_grad():
                self.prediction.bias[: self.num_classes].fill_(bias_value)
            return

        for module in [self.cls_tower, self.reg_tower]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
        # Final projection layers
        for conv in [self.cls_logits, self.box_logits]:
            if conv is not None:
                nn.init.normal_(conv.weight, mean=0.0, std=0.01)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0.0)
        if self.ctr_logits is not None:
            nn.init.normal_(self.ctr_logits.weight, mean=0.0, std=0.01)
            if self.ctr_logits.bias is not None:
                nn.init.constant_(self.ctr_logits.bias, 0.0)
        # Classification bias prior (focal-loss "prior" trick)
        bias_value = math.log(cls_prior / (1.0 - cls_prior))
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_patches, dim = patch_tokens.shape
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError("Patch tokens must form a square feature map.")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)

        if self.architecture == "paper":
            prediction = self.prediction(self.shared_head(feature_map))
            return {
                "class_logits": prediction[:, : self.num_classes],
                "boxes": prediction[:, self.num_classes :],
            }

        out = {
            "class_logits": self.cls_logits(self.cls_tower(feature_map)),
            "boxes": self.box_logits(self.reg_tower(feature_map)),
        }
        if self.use_centerness and self.ctr_logits is not None:
            out["centerness"] = self.ctr_logits(self.reg_tower(feature_map))
        return out


class PROBEModel(nn.Module):
    """End-to-end PROBE scaffold: SPEM-enhanced ViT plus detection head."""

    def __init__(
        self,
        backbone: PromptEnhancedViT,
        detection_head: LightweightDetectionHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.detection_head = detection_head

    def encode(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone(images, prototype_state)

    def detect(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> dict[str, torch.Tensor]:
        _, patch_tokens, _ = self.encode(images, prototype_state)
        return self.detection_head(patch_tokens)
