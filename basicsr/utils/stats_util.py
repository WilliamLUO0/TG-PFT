import csv
import torch
import time
import errno
import numpy as np


def cal_asinh_p90(x, asinh_scale):
    if (asinh_scale is None) or (not np.isfinite(asinh_scale)) or (asinh_scale <= 0):
        return x.astype(np.float32)
    return np.arcsinh(x / float(asinh_scale)).astype(np.float32)


def cal_zscore(x, mean, std):
    return (x - mean) / (std + 1e-12)


def load_npy_shape(path, expect_shape=None, dtype=np.float32, max_retries=7, base_delay=0.4):
    arr = None
    for k in range(max_retries):
        try:
            arr = np.load(path)
            break
        except OSError as e:
            code = getattr(e, "errno", None)
            msg = str(e).lower()
            transient = (code in (121, errno.EIO)) or ("remote i/o" in msg) or ("remote io" in msg) or (
                        "i/o error" in msg)
            if transient and k < max_retries - 1:
                time.sleep(base_delay * (2 ** k))
                continue
            raise
    if expect_shape is not None:
        h, w = expect_shape
        if arr.shape != expect_shape:
            hh = min(h, arr.shape[0])
            ww = min(w, arr.shape[1])
            canvas = np.zeros((h, w), dtype=arr.dtype)
            canvas[:hh, :ww] = arr[:hh, :ww]
            arr = canvas
    return arr.astype(dtype)


def group_by_scenario(rows):
    buckets = {}
    for r in rows:
        buckets.setdefault(r['scenario'], []).append(r)
    return buckets


def load_index_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            r = dict(r)
            r['_row_id'] = i
            rows.append(r)
    if not rows:
        raise RuntimeError(f'[ERROR] index.csv ({path}) has no rows.')
    return rows


def check_required_fields(rows):
    needed = [
        "coarse_path", "fine_path", "elev_fine_path", "elev_coarse_path", "rough_path",
        "mask_fine_path", "mask_coarse_path", "slope_path", "twi_path",
        "aspect_sin_path", "aspect_cos_path", "downscale"
    ]
    missing_any = False
    example_missing = []
    for r in rows:
        for k in needed:
            if k not in r:
                missing_any = True
                example_missing.append(k)
    if missing_any:
        ks = ','.join(sorted(set(example_missing)))
        raise RuntimeError(f'[ERROR] index.csv missing field: {ks}')


def destand_to_physical(z, var, stats_var, h_asinh_q):
    if var == 'h':
        qk = str(int(h_asinh_q))
        S = stats_var['asinh_by_q'][qk]['shared']
        s_val = stats_var['asinh_by_q'][qk]['s']
    else:
        S = stats_var['shared']
        s_key = 'asinh_scale_shared' if ('asinh_scale_shared' in S) else 'asinh_scale'
        s_val = S.get(s_key, 1.0)
        if (s_val is None) or (not np.isfinite(float(s_val))) or (float(s_val) <= 0):
            s_val = 1.0
    mean = torch.tensor(S['mean'], device=z.device, dtype=z.dtype)
    std = torch.tensor(S['std'], device=z.device, dtype=z.dtype)
    s = torch.tensor(float(s_val), device=z.device, dtype=z.dtype)
    return s * torch.sinh(z * std + mean)
