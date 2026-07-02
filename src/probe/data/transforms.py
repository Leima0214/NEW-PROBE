from __future__ import annotations

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DetectionTrainTransform:
    """Detection augmentation that keeps bounding boxes aligned with images."""

    def __init__(self, image_size: int = 512, horizontal_flip_prob: float = 0.5) -> None:
        self.image_size = image_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(
        self,
        image,
        boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, [self.image_size, self.image_size])
        boxes = boxes.clone()

        if torch.rand(()) < self.horizontal_flip_prob:
            image = TF.hflip(image)
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = self.image_size - old_x2
                boxes[:, 2] = self.image_size - old_x1

        image = self.color_jitter(image)
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image, boxes
