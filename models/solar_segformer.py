import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SolarSegFormer(nn.Module):
    """
    SegFormer-based model for Solar Filament Segmentation.
    Optimized for ROCm (uses standard PyTorch ops).
    """
    def __init__(self, model_name="nvidia/mit-b0", num_labels=2):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )

    def forward(self, images, disk_mask=None):
        """
        Args:
            images (torch.Tensor): (B, 3, H, W)
            disk_mask (torch.Tensor): (B, H, W) - 1 for disk, 0 for space.

        Returns:
            logits (torch.Tensor): (B, num_labels, H, W)
        """
        outputs = self.model(pixel_values=images)
        logits = outputs.logits

        # Upsample logits to match image resolution
        # SegFormer outputs are 1/4 of input resolution
        logits = F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits

def weighted_segmentation_loss(logits, targets, disk_mask):
    """
    Custom loss for Solar Segmentation:
    Combines Weighted BCE and Weighted Dice, ignoring 'space' pixels.
    """
    # targets: (B, H, W)
    # logits: (B, C, H, W)
    # disk_mask: (B, H, W)

    # We only care about the 'Filament' class (index 1)
    probs = torch.softmax(logits, dim=1)[:, 1, :, :] # (B, H, W)

    # 1. Weighted BCE
    # BCE per pixel
    bce_loss = F.binary_cross_entropy_with_logits(
        logits[:, 1, :, :],
        targets.float(),
        reduction="none"
    )
    # Apply disk mask
    weighted_bce = (bce_loss * disk_mask).sum() / disk_mask.sum().clamp(min=1)

    # 2. Weighted Dice Loss
    # intersection = (prob * target * mask).sum()
    intersection = (probs * targets * disk_mask).sum()
    union = (probs * disk_mask).sum() + (targets * disk_mask).sum()

    dice_loss = 1 - (2. * intersection + 1) / (union + 1)

    return weighted_bce + dice_loss
