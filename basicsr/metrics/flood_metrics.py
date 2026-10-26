from typing import Optional

import torch

from basicsr.utils.registry import METRIC_REGISTRY


def ensure_4d_pt(x: torch.Tensor):
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    return x


def sum_over_hw(x: torch.Tensor):
    return x.flatten(1).sum(1)


def _safe_divide_metric_pt(num: torch.Tensor,
                           denom: torch.Tensor,
                           eps: float = 1e-12,
                           empty_as_nan: bool = False):
    if empty_as_nan:
        out = torch.full_like(num, float("nan"))
        valid = denom > 0
        out[valid] = num[valid] / denom[valid].clamp_min(eps)
        return out

    return num / denom.clamp_min(eps)


def _reduce_metric_pt(x: torch.Tensor, reduction: str = "mean"):
    if reduction == "none":
        return x

    finite = torch.isfinite(x)
    if finite.any():
        return x[finite].mean().item()
    return float("nan")


@METRIC_REGISTRY.register()
def cal_rmse_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    diff2 = (pred - target) ** 2
    masked_diff2 = sum_over_hw(diff2 * mask)
    num_mask = sum_over_hw(mask).clamp_min(eps)

    rmse = torch.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return rmse.mean().item()


@METRIC_REGISTRY.register()
def cal_rmse_conditional_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                            reduction: str = "mean", eps: float = 1e-12,
                            cond_on_target_ge: Optional[float] = None,
                            cond_on_target_lt: Optional[float] = None,
                            abs_tol: Optional[float] = None,
                            use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    sel = m

    sel_target = target.abs() if use_abs else target

    if cond_on_target_ge is not None:
        sel = sel & (sel_target >= cond_on_target_ge)

    if cond_on_target_lt is not None:
        sel = sel & (sel_target < cond_on_target_lt)

    diff = pred - target
    if abs_tol is not None:
        diff = torch.where(diff.abs() < abs_tol, torch.zeros_like(diff), diff)

    diff2 = diff ** 2
    sse = sum_over_hw(diff2 * sel.to(diff2.dtype))
    n = sum_over_hw(sel.to(diff2.dtype))
    valid = n > 0.5

    rmse = torch.empty_like(sse)
    rmse[valid] = torch.sqrt(sse[valid] / n[valid].clamp_min(eps))
    rmse[~valid] = torch.tensor(float("nan"), device=rmse.device, dtype=rmse.dtype)

    if reduction == "none":
        return rmse

    if valid.any():
        return rmse[valid].mean().item()
    else:
        return float("nan")


@METRIC_REGISTRY.register()
def cal_nse_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask
    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    masked_diff2 = sum_over_hw(((pred - target) ** 2) * w)
    vc = sum_over_hw(((target - t_mean) ** 2) * w)
    nse = 1.0 - (masked_diff2 / vc.clamp_min(eps))
    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_csi_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               threshold: float = 0.05, reduction: str = "mean",
               empty_as_nan: bool = False, eps: float = 1e-12,
               use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if use_abs:
        pred = pred.abs()
        target = target.abs()

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    csi = _safe_divide_metric_pt(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     threshold: float = 0.05, reduction: str = "mean",
                     empty_as_nan: bool = False, eps: float = 1e-12,
                     use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if use_abs:
        pred = pred.abs()
        target = target.abs()

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                          band_ge: Optional[float] = None,
                          band_lt: Optional[float] = None,
                          reduction: str = "mean",
                          empty_as_nan: bool = False,
                          eps: float = 1e-12,
                          use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if use_abs:
        pred = pred.abs()
        target = target.abs()

    m = mask > 0.5

    p_evt = m
    t_evt = m

    if band_ge is not None:
        p_evt = p_evt & (pred >= band_ge)
        t_evt = t_evt & (target >= band_ge)

    if band_lt is not None:
        p_evt = p_evt & (pred < band_lt)
        t_evt = t_evt & (target < band_lt)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  threshold: float = 0.05, reduction: str = "mean",
                  empty_as_nan: bool = False, eps: float = 1e-12,
                  use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if use_abs:
        pred = pred.abs()
        target = target.abs()

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(rec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                       band_ge: Optional[float] = None,
                       band_lt: Optional[float] = None,
                       reduction: str = "mean",
                       empty_as_nan: bool = False,
                       eps: float = 1e-12,
                       use_abs: bool = False):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if use_abs:
        pred = pred.abs()
        target = target.abs()

    m = mask > 0.5

    p_evt = m
    t_evt = m

    if band_ge is not None:
        p_evt = p_evt & (pred >= band_ge)
        t_evt = t_evt & (target >= band_ge)

    if band_lt is not None:
        p_evt = p_evt & (pred < band_lt)
        t_evt = t_evt & (target < band_lt)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(rec, reduction=reduction)
