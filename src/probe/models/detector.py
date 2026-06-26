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
    """FCOS-style detection head with centerness (standard practice).

    Shared stem (paper-aligned) + three parallel branches:
      - Classification: C sigmoid logits per location
      - Box regression: 4 [l, t, r, b] distances per location
      - Centerness:     1 score per location (suppresses edge boxes)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        neck_dim: int = 128,
        num_classes: int = 5,
        cls_prior: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.cls_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, num_classes, kernel_size=1),
        )
        self.box_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, 4, kernel_size=1),
        )
        self.ctr_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, 1, kernel_size=1),
        )

        self._init_weights(cls_prior)

    def _init_weights(self, cls_prior: float = 0.01) -> None:
        for branch in [self.cls_branch, self.box_branch, self.ctr_branch]:
            last_conv = branch[-1]
            if isinstance(last_conv, nn.Conv2d):
                nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
                if last_conv.bias is not None:
                    nn.init.constant_(last_conv.bias, 0.0)
        # Classification bias prior
        last_cls = self.cls_branch[-1]
        if isinstance(last_cls, nn.Conv2d) and last_cls.bias is not None:
            bias_value = math.log(cls_prior / (1.0 - cls_prior))
            nn.init.constant_(last_cls.bias, bias_value)

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_patches, dim = patch_tokens.shape
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError("Patch tokens must form a square feature map.")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)

        shared = self.stem(feature_map)
        return {
            "class_logits": self.cls_branch(shared),
            "boxes": self.box_branch(shared),
            "centerness": self.ctr_branch(shared),
        }


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
