import os
import json
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data as data

from basicsr.utils import (
    cal_zscore, cal_asinh_p90, load_npy_shape, load_index_csv,
    check_required_fields, get_root_logger
)
from basicsr.data.transforms import augment_flood_map
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedFloodMapDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedFloodMapDataset, self).__init__()
        self.opt = deepcopy(opt)

        self.phase = opt.get('phase', 'train')
        assert self.phase in ('train', 'val', 'test'), f'phase must be train/val/test, got {self.phase}'

        self.target_var = opt.get('target_var', 'h')
        assert self.target_var in ('h', 'u', 'v'), f"target_var should be one of ['h', 'u', 'v'], got {self.target_var}"

        self.aux_vars = self._parse_aux_vars(opt.get('aux_vars', []))
        allowed_aux = {'zs', 'dem'}
        unknown_aux = [x for x in self.aux_vars if x not in allowed_aux]
        if len(unknown_aux) > 0:
            raise ValueError(f'[ERROR] Unsupported aux_vars: {unknown_aux}, allowed: {sorted(list(allowed_aux))}')

        if self.target_var in ('u', 'v') and 'zs' in self.aux_vars:
            raise ValueError(
                f"[ERROR] aux var 'zs' is not allowed for target_var={self.target_var} "
                f"(it leaks water-depth information). Only 'dem' is allowed as a coarse "
                f"aux for u/v. Got aux_vars={self.aux_vars}"
            )

        self.static_vars = ['dem', 'roughness', 'slope', 'twi', 'aspect_sin', 'aspect_cos', 'mask']

        self.scale = int(opt.get('scale', 16))
        self.patch_fine = int(opt.get('patch_fine', 1024))
        self.patch_coarse = int(opt.get('patch_coarse', 64))

        self.root = opt['dataroot']
        self.index_csv = opt.get('index_csv', os.path.join(self.root, 'index.csv'))
        if not os.path.isfile(self.index_csv):
            raise FileNotFoundError(f'[ERROR] index_csv not found: {self.index_csv}')

        self.use_hflip = bool(opt.get('use_hflip', False))
        self.use_rot = bool(opt.get('use_rot', False))

        split_cfg = opt.get('split_cfg', {}) or {}
        self.split_by = split_cfg.get('by', 'scenario')
        self.val_ratio = float(split_cfg.get('val_ratio', 0.2))
        self.seed = int(split_cfg.get('seed', 61))
        self.split_stats_json = split_cfg.get(
            'split_stats_json',
            os.path.join(self.root, 'dataset', f'split_stats_{self.target_var}.json')
        )

        stats_cfg = opt.get('stats', {}) or {}
        self.h_asinh_q = int(stats_cfg.get('h_asinh_q', 90))

        all_rows = load_index_csv(self.index_csv)
        rows = [r for r in all_rows if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == self.target_var]

        self.meta = self._ensure_split_stats(var=self.target_var, aux_vars=self.aux_vars)
        self.stats_static = self.meta['stats']
        self.stats_var = self.meta['stats_var']
        self.aux_stats = self.meta.get('aux_stats', {}) or {}

        if self.phase in ('train', 'val'):
            ids = set(self.meta['split'][self.phase])
            self.rows = [r for r in rows if int(r['_row_id']) in ids]
        elif self.phase == 'test':
            self.rows = rows
        else:
            raise ValueError(f'[ERROR] Unknown phase: {self.phase}')

        check_required_fields(self.rows)
        self._check_required_fields_for_aux(self.rows)

        coarse_channels_log = [f'{self.target_var}(coarse_flood_map)'] + list(self.aux_vars)
        n_coarse = 1 + len(self.aux_vars)
        get_root_logger().info(
            f'[PairedFloodMapDataset] phase={self.phase} INPUTS:\n'
            f'  - coarse_flood_map ({n_coarse} ch -> set network_g.coarse_in_chans='
            f'{n_coarse}): {coarse_channels_log}\n'
            f'  - fine_static_feature ({len(self.static_vars)} ch -> set network_g.static_in_chans='
            f'{len(self.static_vars)}, mask is last): {self.static_vars}\n'
            f'  - target: fine_flood_map({self.target_var})'
        )

    def _parse_aux_vars(self, aux_raw):
        if aux_raw is None:
            return []

        if isinstance(aux_raw, str):
            s = aux_raw.strip()
            if s == '':
                return []
            return [x.strip() for x in s.split(',') if x.strip() != '']

        if isinstance(aux_raw, (list, tuple, set)):
            return [str(x).strip() for x in aux_raw if str(x).strip() != '']

        raise TypeError(f'[ERROR] aux_vars must be str/list/tuple/set, got {type(aux_raw)}')

    def _check_required_fields_for_aux(self, rows):
        if len(rows) == 0:
            return

        if 'zs' in self.aux_vars:
            if 'zs_coarse_path' not in rows[0]:
                raise RuntimeError('[ERROR] index.csv missing field for aux var zs: zs_coarse_path')

        if 'dem' in self.aux_vars:
            if 'elev_coarse_path' not in rows[0]:
                raise RuntimeError('[ERROR] index.csv missing field for aux var dem: elev_coarse_path')

    def _ensure_split_stats(self, var: str, aux_vars=None):
        aux_vars = aux_vars or []

        if not os.path.isfile(self.split_stats_json):
            raise FileNotFoundError(
                f'[ERROR] split_stats_json not found: {self.split_stats_json}\n'
                f'Pre-compute it via scripts/data/precompute_split_stats.py before training.'
            )

        with open(self.split_stats_json, 'r') as f:
            meta = json.load(f)

        for k in ('split', 'seed', 'val_ratio', 'stats', 'stats_var'):
            if k not in meta:
                raise RuntimeError(
                    f'[ERROR] split_stats_json (path: {self.split_stats_json}) missing key: {k}. '
                    f'Re-generate the JSON via scripts/data/precompute_split_stats.py.'
                )

        if meta.get('stats_var_for', None) != var:
            raise RuntimeError(
                f'[ERROR] split_stats_json (path: {self.split_stats_json}) has '
                f'stats_var_for={meta.get("stats_var_for")}, expected {var}. '
                f'Re-generate the JSON for var={var}.'
            )

        if 'zs' in aux_vars:
            aux_stats = meta.get('aux_stats', {}) or {}
            if 'zs' not in aux_stats:
                raise RuntimeError(
                    f'[ERROR] aux_vars includes "zs", but split_stats_json does not contain '
                    f'meta["aux_stats"]["zs"]. Please regenerate stats json with --aux_vars zs'
                )

        return meta

    def __len__(self):
        return len(self.rows)

    def get_severity_sample_weights(self, sampler_cfg=None):
        sampler_cfg = sampler_cfg or {}

        base_weight = float(sampler_cfg.get('base_weight', 1.0))
        max_weight = float(sampler_cfg.get('max_weight', 10.0))

        slight_alpha = float(sampler_cfg.get('slight_alpha', 4.0))
        severe_alpha = float(sampler_cfg.get('severe_alpha', 2.0))
        extreme_alpha = float(sampler_cfg.get('extreme_alpha', 2.0))

        eps = 1e-12

        def _to_float(x, default=0.0):
            if x is None:
                return default
            if isinstance(x, str) and x.strip() == '':
                return default
            return float(x)

        def _band_score(ratio, low, high):
            denom = max(high - low, eps)
            score = (ratio - low) / denom
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return float(score)

        def _load_percentile_refs_from_summary(summary_json, percentile_source, low_percentile, high_percentile):
            if summary_json is None or str(summary_json).strip() == '':
                raise ValueError(
                    '[ERROR] sampler.summary_json is required.'
                )

            summary_json = str(summary_json)
            if not os.path.isfile(summary_json):
                raise FileNotFoundError(f'[ERROR] sampler.summary_json not found: {summary_json}')

            with open(summary_json, 'r') as f:
                summary = json.load(f)

            if percentile_source == 'positive':
                key = 'ratio_percentiles_positive_patches'
            elif percentile_source == 'all':
                key = 'ratio_percentiles_all_patches'
            else:
                raise ValueError(
                    f"[ERROR] percentile_source must be 'positive' or 'all', got {percentile_source}"
                )

            if key not in summary:
                raise RuntimeError(f'[ERROR] summary_json missing key: {key}')

            refs = {}

            for name in ['slight_ratio', 'severe_ratio', 'extreme_ratio']:
                if name not in summary[key]:
                    raise RuntimeError(f'[ERROR] summary_json missing {key}/{name}')

                low = summary[key][name].get(low_percentile, None)
                high = summary[key][name].get(high_percentile, None)

                if low is None or high is None:
                    raise RuntimeError(
                        f'[ERROR] summary_json missing percentile for {name}: '
                        f'{low_percentile}={low}, {high_percentile}={high}'
                    )

                low = float(low)
                high = float(high)

                if not np.isfinite(low) or not np.isfinite(high):
                    raise RuntimeError(
                        f'[ERROR] invalid percentile values for {name}: low={low}, high={high}'
                    )

                if high <= low:
                    raise RuntimeError(
                        f'[ERROR] high percentile must be larger than low percentile for {name}: '
                        f'{high_percentile}={high}, {low_percentile}={low}'
                    )

                refs[name] = {
                    'low': low,
                    'high': high,
                }

            return refs

        weights = []

        refs = _load_percentile_refs_from_summary(
            summary_json=sampler_cfg.get('summary_json', ''),
            percentile_source=str(sampler_cfg.get('percentile_source', 'positive')).lower().strip(),
            low_percentile=str(sampler_cfg.get('low_percentile', 'p75')).strip(),
            high_percentile=str(sampler_cfg.get('high_percentile', 'p90')).strip(),
        )

        for r in self.rows:
            slight_ratio = _to_float(r.get('slight_ratio', 0.0))
            severe_ratio = _to_float(r.get('severe_ratio', 0.0))
            extreme_ratio = _to_float(r.get('extreme_ratio', 0.0))

            slight_score = _band_score(slight_ratio, refs['slight_ratio']['low'], refs['slight_ratio']['high'])
            severe_score = _band_score(severe_ratio, refs['severe_ratio']['low'], refs['severe_ratio']['high'])
            extreme_score = _band_score(extreme_ratio, refs['extreme_ratio']['low'], refs['extreme_ratio']['high'])

            w = base_weight
            w += slight_alpha * slight_score
            w += severe_alpha * severe_score
            w += extreme_alpha * extreme_score
            w = min(w, max_weight)

            weights.append(w)

        weights = torch.as_tensor(weights, dtype=torch.double)

        if weights.numel() != len(self.rows):
            raise RuntimeError(
                f'[ERROR] weight number mismatch: weights={weights.numel()}, rows={len(self.rows)}'
            )

        if torch.any(~torch.isfinite(weights)):
            raise RuntimeError('[ERROR] severity sample weights contain NaN or Inf.')

        if torch.any(weights < 0):
            raise RuntimeError('[ERROR] severity sample weights contain negative values.')

        if torch.sum(weights) <= 0:
            raise RuntimeError('[ERROR] sum of severity sample weights <= 0.')

        return weights

    def __getitem__(self, index):
        r = self.rows[index]
        Hf = self.patch_fine
        Hc = self.patch_coarse

        coarse_fm = load_npy_shape(r['coarse_path'], expect_shape=(Hc, Hc))
        if (self.phase == 'test') and (not os.path.isfile(r.get('fine_path', ''))):
            if not getattr(self, '_warned_missing_fine_path', False):
                get_root_logger().warning(
                    f'[PairedFloodMapDataset] phase=test: fine_path missing for '
                    f'row scenario={r.get("scenario")}, t={r.get("t")}, '
                    f'patch=({r.get("patch_row")},{r.get("patch_col")}). '
                    f'Returning zero fine_fm. Eval metrics on these patches '
                    f'will be invalid; only use predictions for inference.'
                )
                self._warned_missing_fine_path = True
            fine_fm = np.zeros((Hf, Hf), dtype=np.float32)
        else:
            fine_fm = load_npy_shape(r['fine_path'], expect_shape=(Hf, Hf))

        elev_key = 'elev_fine_path' if 'elev_fine_path' in r else 'elev_path'
        elev = load_npy_shape(r[elev_key], expect_shape=(Hf, Hf))
        rough = load_npy_shape(r['rough_path'], expect_shape=(Hf, Hf))
        slope = load_npy_shape(r['slope_path'], expect_shape=(Hf, Hf))
        twi = load_npy_shape(r['twi_path'], expect_shape=(Hf, Hf))
        asin = load_npy_shape(r['aspect_sin_path'], expect_shape=(Hf, Hf))
        acos = load_npy_shape(r['aspect_cos_path'], expect_shape=(Hf, Hf))
        mask_fine = load_npy_shape(r['mask_fine_path'], expect_shape=(Hf, Hf), dtype=np.uint8)

        aux_loaded = {}
        for aux_name in self.aux_vars:
            if aux_name == 'zs':
                aux_loaded['zs'] = load_npy_shape(r['zs_coarse_path'], expect_shape=(Hc, Hc))
            elif aux_name == 'dem':
                aux_loaded['dem'] = load_npy_shape(r['elev_coarse_path'], expect_shape=(Hc, Hc))
            else:
                raise ValueError(f'[ERROR] Unsupported aux var: {aux_name}')

        transform_list = [
            coarse_fm,
            fine_fm,
            elev,
            rough,
            slope,
            twi,
            asin,
            acos,
            mask_fine.astype(np.float32)
        ]

        aux_transform_indices = {}
        for aux_name in self.aux_vars:
            aux_transform_indices[aux_name] = len(transform_list)
            transform_list.append(aux_loaded[aux_name])

        transform_list = augment_flood_map(
            transform_list,
            target_var=self.target_var,
            use_hflip=self.use_hflip,
            use_rot=self.use_rot,
            is_train_phase=(self.phase == 'train'),
            idx_coarse=0,
            idx_fine=1,
            idx_asin=6,
            idx_acos=7
        )

        coarse_fm = transform_list[0]
        fine_fm = transform_list[1]
        elev = transform_list[2]
        rough = transform_list[3]
        slope = transform_list[4]
        twi = transform_list[5]
        asin = transform_list[6]
        acos = transform_list[7]
        mask_fine = (transform_list[8] > 0.5).astype(np.float32)

        for aux_name in self.aux_vars:
            aux_loaded[aux_name] = transform_list[aux_transform_indices[aux_name]]

        S_static = self.stats_static
        S_var = self.stats_var

        if self.target_var == 'h':
            asinh_stats = S_var['asinh_by_q'][str(self.h_asinh_q)]
            S_shared = asinh_stats['shared']
            s_val = float(asinh_stats['s'])
        else:
            S_shared = S_var['shared']
            s_val = float(S_shared['asinh_scale_shared'])
        cfm = cal_zscore(cal_asinh_p90(coarse_fm, s_val), S_shared['mean'], S_shared['std'])
        ffm = cal_zscore(cal_asinh_p90(fine_fm, s_val), S_shared['mean'], S_shared['std'])

        e = cal_zscore(elev, S_static['elevation']['mean'], S_static['elevation']['std'])
        rgh = cal_zscore(rough, S_static['roughness']['mean'], S_static['roughness']['std'])
        tf = cal_zscore(twi, S_static['twi']['mean'], S_static['twi']['std'])
        slp = slope / 90.0

        fine_static_feature = np.stack([e, rgh, slp, tf, asin, acos, mask_fine], axis=0).astype(np.float32)

        coarse_channels = [cfm.astype(np.float32)]

        for aux_name in self.aux_vars:
            if aux_name == 'zs':
                if 'zs' not in self.aux_stats:
                    raise RuntimeError('[ERROR] aux_stats["zs"] not found in split_stats_json.')

                S_zs = self.aux_stats['zs']['shared']
                arr = cal_zscore(aux_loaded['zs'], S_zs['mean'], S_zs['std'])
                coarse_channels.append(arr.astype(np.float32))

            elif aux_name == 'dem':
                S_elev = S_static['elevation']
                arr = cal_zscore(aux_loaded['dem'], S_elev['mean'], S_elev['std'])
                coarse_channels.append(arr.astype(np.float32))

            else:
                raise ValueError(f'[ERROR] Unsupported aux var during normalization: {aux_name}')

        coarse_flood_map = np.stack(coarse_channels, axis=0).astype(np.float32)
        fine_flood_map = np.expand_dims(ffm.astype(np.float32), axis=0)

        sample = {
            'coarse_flood_map': torch.from_numpy(coarse_flood_map),
            'fine_static_feature': torch.from_numpy(fine_static_feature),
            'fine_flood_map': torch.from_numpy(fine_flood_map),
            'meta': {
                'scenario': r['scenario'],
                't': r['t'],
                'downscale': self.scale,
                'row': int(r['patch_row']),
                'col': int(r['patch_col']),
                'var': self.target_var,
                'aux_vars': list(self.aux_vars),
                'coarse_path': r['coarse_path'],
                'fine_path': r['fine_path'],
            }
        }

        return sample
