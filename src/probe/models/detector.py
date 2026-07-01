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
    ) -> None:
        super().__init__()
        self.vit = vit
        self.prompt_projector = prompt_projector
        self.injector = PromptInjector(injection_layers)
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

        for layer_id, block in enumerate(self.vit.blocks):
            if self.injector.should_inject(layer_id):
                K = prompts.shape[1]
                tokens = self.injector.insert(tokens, prompts)
                tokens = block(tokens)
                processed_prompts = tokens[:, :K, :]
                tokens = self.injector.remove(tokens, K)
            else:
                tokens = block(tokens)

        return self.vit.norm(tokens), prompts, processed_prompts

    def forward(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, _, processed_prompts = self.forward_tokens(images, prototype_state)
        image_features = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return image_features, patch_tokens, processed_prompts


class LightweightDetectionHead(nn.Module):
    """FCOS detection head with separate classification and regression towers.

    Standard FCOS practice: each tower is a stack of 3×3 convolutions with
    GroupNorm (more stable than BatchNorm for cross-domain settings).
    The classification tower outputs C sigmoid logits per location; the
    regression tower is shared by the box (4-d) and centerness (1-d) branches.
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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_centerness = use_centerness

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
