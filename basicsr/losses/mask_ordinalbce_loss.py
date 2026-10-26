import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskOrdinalBCELoss(nn.Module):

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau1_m: float = 0.1,
        tau2_m: float = 0.5,
        tau3_m: float = 1.0,
        abs_target: bool = False,
        var: str = "h",
        stats_json: str = "",
        h_asinh_q: int = 90,
        pos_weight_min: float = 0.5,
        pos_weight_max: float = 50.0,
        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.tau1_m = float(tau1_m)
        self.tau2_m = float(tau2_m)
        self.tau3_m = float(tau3_m)
        self.abs_target = bool(abs_target)
        self.var = str(var).lower().strip()
        self.h_asinh_q = int(h_asinh_q)
        self.pos_weight_min = float(pos_weight_min)
        self.pos_weight_max = float(pos_weight_max)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskOrdinalBCELoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskOrdinalBCELoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskOrdinalBCELoss] stats_json must contain key 'stats_var': {stats_json}")
        S_var = meta["stats_var"]

        if self.var == "h":
            node = S_var["asinh_by_q"][str(self.h_asinh_q)]
            self.asinh_scale = float(node["s"])
            S = node["shared"]
        else:
            S = S_var["shared"]
            self.asinh_scale = float(S.get("asinh_scale_shared", 1.0))

        self.stats_mean = float(S["mean"])
        self.stats_std = float(S["std"])

        tau_t = [self._asinh_scalar(t) for t in (self.tau1_m, self.tau2_m, self.tau3_m)]
        tau_std = [(t - self.stats_mean) / (self.stats_std + self.eps) for t in tau_t]

        self.register_buffer("tau1_std", torch.tensor(tau_std[0], dtype=torch.float32))
        self.register_buffer("tau2_std", torch.tensor(tau_std[1], dtype=torch.float32))
        self.register_buffer("tau3_std", torch.tensor(tau_std[2], dtype=torch.float32))

        if self.abs_target:
            self.register_buffer("tau1_t", torch.tensor(tau_t[0], dtype=torch.float32))
            self.register_buffer("tau2_t", torch.tensor(tau_t[1], dtype=torch.float32))
            self.register_buffer("tau3_t", torch.tensor(tau_t[2], dtype=torch.float32))

        logger.info(
            f"[MaskOrdinalBCELoss] var={self.var}, abs_target={self.abs_target}, "
            f"tau_std=({tau_std[0]:.6f}, {tau_std[1]:.6f}, {tau_std[2]:.6f}), "
            f"pos_weight_clip=[{self.pos_weight_min}, {self.pos_weight_max}]"
        )

    def _asinh_scalar(self, x_m: float) -> float:
        return math.asinh(float(x_m) / max(self.asinh_scale, 1e-12))

    def _masked_binary_loss(self, logit, target, mask):
        pos = (target * mask).sum()
        neg = ((1.0 - target) * mask).sum()
        pw = (neg / (pos + self.eps)).clamp(self.pos_weight_min, self.pos_weight_max)
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none", pos_weight=pw)

        masked = bce * mask
        per_sample_num = mask.flatten(1).sum(1)
        per_sample_loss = masked.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return logit.new_tensor(0.0)
            return (per_sample_loss[valid] / (per_sample_num[valid] + self.eps)).mean()
        return (per_sample_loss / (per_sample_num + self.eps)).mean()

    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        if pred_logit.dim() != 4 or pred_logit.shape[1] != 3:
            raise ValueError(f"[MaskOrdinalBCELoss] pred_logit must have shape [B,3,H,W], got {pred_logit.shape}")

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred_logit.dtype)

        with torch.no_grad():
            if self.abs_target:
                av = (target_depth * (self.stats_std + self.eps) + self.stats_mean).abs()
                t1 = (av >= self.tau1_t).to(dtype=pred_logit.dtype)
                t2 = (av >= self.tau2_t).to(dtype=pred_logit.dtype)
                t3 = (av >= self.tau3_t).to(dtype=pred_logit.dtype)
            else:
                t1 = (target_depth >= self.tau1_std).to(dtype=pred_logit.dtype)
                t2 = (target_depth >= self.tau2_std).to(dtype=pred_logit.dtype)
                t3 = (target_depth >= self.tau3_std).to(dtype=pred_logit.dtype)

        l1 = self._masked_binary_loss(pred_logit[:, 0:1], t1, mask)
        l2 = self._masked_binary_loss(pred_logit[:, 1:2], t2, mask)
        l3 = self._masked_binary_loss(pred_logit[:, 2:3], t3, mask)

        return self.loss_weight * (l1 + l2 + l3)
