import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskL1Loss(nn.Module):

    def __init__(self, loss_weight: float = 1.0, ignore_zero_mask: bool = True, eps: float = 1e-12):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        diff = F.l1_loss(pred, target, reduction="none")
        masked = diff * mask

        per_sample_num = mask.flatten(1).sum(1)
        per_sample_loss = masked.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return pred.new_tensor(0.0)
            loss = (per_sample_loss[valid] / (per_sample_num[valid] + self.eps)).mean()
        else:
            loss = (per_sample_loss / (per_sample_num + self.eps)).mean()

        return self.loss_weight * loss
