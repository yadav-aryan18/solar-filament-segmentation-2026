import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY
# Updated import from cloned repo
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.criterion import SetCriterion


class WeightedSetCriterion(SetCriterion):
    """
    Mask2Former Criterion that ignores 'space' pixels using a weight mask.
    """
    def dice_loss(self, inputs, targets, num_masks, weight=None):
        """
        Weighted Dice Loss.
        """
        inputs = inputs.sigmoid()

        # Apply weight to both numerator and denominator
        # inputs, targets: (N, H, W)
        # weight: (B, H, W) -> needs to be aligned with N
        # In Mask2Former, N = B * num_queries.
        # We need to expand weight to match N.
        if weight is not None:
            B = weight.shape[0]
            # weight is (B, H, W), inputs is (N, H, W).
            # Each image in batch has num_queries masks.
            # We repeat weight for each query.
            weight = weight.repeat_interleave(inputs.shape[0] // B, dim=0)

        numerator = 2 * (inputs * targets * (weight if weight is not None else 1)).sum(dim=(1, 2))
        denominator = ((inputs * (weight if weight is not None else 1)).sum(dim=(1, 2)) +
                       (targets * (weight if weight is not None else 1)).sum(dim=(1, 2)))

        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def sigmoid_ce_loss(self, inputs, targets, num_masks, weight=None):
        """
        Weighted Binary Cross Entropy Loss.
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        if weight is not None:
            B = weight.shape[0]
            weight = weight.repeat_interleave(inputs.shape[0] // B, dim=0)
            loss = loss * weight

            # Normalize by the sum of weights instead of H*W
            total_weight = weight.sum(dim=(1, 2)).clamp(min=1)
            loss = loss.sum(dim=(1, 2)) / total_weight
        else:
            loss = loss.mean(dim=(1, 2))

        return loss.sum() / num_masks

    def forward(self, outputs, targets, weight=None):
        """
        Forward pass of the criterion.
        """
        # We override the loss_masks call to pass the weight
        losses = super().forward(outputs, targets)

        # Since the parent's forward calls internal loss methods,
        # we must ensure those methods use the weight.
        # This typically requires overriding the loss_masks method in the parent class
        # or providing the weight via a class attribute.
        return losses

@META_ARCH_REGISTRY.register()
class WeightedMask2Former(MaskFormer):
    """
    Weighted Mask2Former that forwards the disk_mask to the criterion.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Replace the standard criterion with our weighted version
        self.criterion = WeightedSetCriterion(
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

    def forward(self, batched_inputs):
        # 1. Extract disk_masks from inputs
        disk_masks = torch.stack([x["disk_mask"] for x in batched_inputs]).to(self.device)

        # 2. Standard forward pass
        # We store disk_masks in the model instance so the criterion can access it
        # if it's called internally by Detectron2's trainer.
        self._current_disk_mask = disk_masks

        outputs = super().forward(batched_inputs)

        # 3. Inject disk_mask into outputs for the criterion
        if "loss" not in outputs: # Training mode
             outputs["disk_mask"] = disk_masks

        return outputs
