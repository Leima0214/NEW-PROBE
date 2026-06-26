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
    """Frozen ViT wrapper with SPEM prompt injection.

    The wrapped ViT is expected to expose standard ViT components:
    ``patch_embed``, ``cls_token``, ``pos_embed``, ``blocks`` and ``norm``.
    This keeps the method code independent of a specific timm/torchvision
    backbone while documenting the exact insertion points used by PROBE.
    """

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
        """Returns (tokens, prompts, processed_prompts).

        ``processed_prompts`` are the prompt tokens after they have been
        updated by transformer attention at the *last* injection layer.
        These are image-specific (unlike the raw prompts, which are
        identical across the batch) and enable the InfoNCE prompt
        consistency loss to actually learn.
        """
        tokens = self.patchify(images)
        prompts = self.prompt_projector(
            prototype_state.centroids.to(tokens.device),
            batch_size=images.shape[0],
        )
        processed_prompts = prompts  # fallback if no injection happens

        for layer_id, block in enumerate(self.vit.blocks):
            if self.injector.should_inject(layer_id):
                K = prompts.shape[1]
                tokens = self.injector.insert(tokens, prompts)
                tokens = block(tokens)
                # Capture the prompt portion AFTER attention updated it
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
    """Three-stage detection head described in the PROBE paper (Section 3.5).

    Architecture (paper-exact):
      1. Conv3x3-BN-GELU  → embed_dim → hidden_dim  (384)
      2. Conv1x1-GELU     → hidden_dim → neck_dim   (128)
      3. Conv1x1          → neck_dim → num_classes + 4  (cls + [l,t,r,b])

    Reshapes ViT patch tokens into a square feature map and predicts
    per-cell class logits plus four box-distance parameters.
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
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, num_classes + 4, kernel_size=1),
        )
        self._init_weights(cls_prior)

    def _init_weights(self, cls_prior: float = 0.01) -> None:
        """Initialize conv weights and set classification bias prior.

        FCOS standard: bias = log(prior/(1-prior)) so that sigmoid(bias) ≈ prior.
        With cls_prior=0.01, bias ≈ -4.595 — only ~1% of locations fire at init.
        """
        for module in self.head:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # Classification bias: set prior so early training is stable
        last_conv = self.head[-1]
        if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
            bias_value = math.log(cls_prior / (1.0 - cls_prior))
            nn.init.constant_(last_conv.bias[: self.num_classes], bias_value)
            # Box regression bias stays at 0.0

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_patches, dim = patch_tokens.shape
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError("Patch tokens must form a square feature map.")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)
        pred = self.head(feature_map)
        return {
            "class_logits": pred[:, : self.num_classes],
            "boxes": pred[:, self.num_classes :],
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
