import os
import argparse
import numpy as np
import xarray as xr

try:
    import rioxarray as rxr
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False

try:
    import richdem as rd
    _HAS_RICHDEM = True
except Exception:
    _HAS_RICHDEM = False


FILL_SENTINEL = np.float32(9.96921e36)
RD_NODATA     = -1.0e6


def horn_attrs(z, cs, y_ascending):
    ny, nx = z.shape
    slope_deg = np.full((ny, nx), np.nan)
    asin = np.full((ny, nx), np.nan)
    acos = np.full((ny, nx), np.nan)

    a = z[:-2, :-2]; b = z[:-2, 1:-1]; c = z[:-2, 2:]
    d = z[1:-1, :-2];                  f = z[1:-1, 2:]
    g = z[2:,  :-2]; h = z[2:,  1:-1]; i = z[2:,  2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cs)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cs)
    if not y_ascending:
        dz_dy = -dz_dy

    slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    azimuth = np.arctan2(-dz_dx, -dz_dy)
    s = np.sin(azimuth)
    co = np.cos(azimuth)
    flat = slope == 0
    s = np.where(flat, 0.0, s)
    co = np.where(flat, 0.0, co)

    slope_deg[1:-1, 1:-1] = slope
    asin[1:-1, 1:-1] = s
    acos[1:-1, 1:-1] = co
    return slope_deg, asin, acos


def slope_radians_from(z, cs):
    ny, nx = z.shape
    out = np.full((ny, nx), np.nan)
    a = z[:-2, :-2]; b = z[:-2, 1:-1]; c = z[:-2, 2:]
    d = z[1:-1, :-2];                  f = z[1:-1, 2:]
    g = z[2:,  :-2]; h = z[2:,  1:-1]; i = z[2:,  2:]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cs)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * cs)
    out[1:-1, 1:-1] = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    return out


def read_geotransform(ds, var, cs):
    gt = None
    for src in (ds[var].attrs, ds.attrs):
        if "GeoTransform" in src:
            try:
                gt = [float(t) for t in str(src["GeoTransform"]).split()]
            except Exception:
                gt = None
    return gt if (gt and len(gt) == 6) else [0.0, cs, 0.0, 0.0, 0.0, -cs]


def compute_twi(z, valid, cs, geotransform, verbose=True):
    if not _HAS_RICHDEM:
        raise RuntimeError("richdem is required for the wetness index")
    zmax = float(np.nanmax(z))
    z_rd = np.where(valid, z, RD_NODATA).astype(np.float64)
    dem = rd.rdarray(z_rd, no_data=RD_NODATA)
    dem.geotransform = geotransform

    try:
        dem_filled = rd.FillDepressions(dem, epsilon=True, in_place=False)
    except TypeError:
        dem_filled = rd.FillDepressions(dem, in_place=False)

    filled = np.asarray(dem_filled, dtype=np.float64)
    rise = filled[valid] - z[valid]
    if verbose:
        print(f"[check] fill rise over valid cells: max={np.nanmax(rise):.3f} m  "
              f"mean={np.nanmean(rise):.4f} m  cells raised = {100.0*np.mean(rise > 0):.2f}%")
    if np.nanmax(filled[valid]) > zmax + 1e4:
        raise SystemExit("[fatal] the filled surface exceeds the terrain by more than 10 km; "
                         "the no-data sentinel was not honoured")

    acc_cells = rd.FlowAccumulation(dem_filled, method='D8')
    acc = np.asarray(acc_cells, dtype=np.float64)
    big = float(np.mean(acc[valid] >= 1000))
    if verbose:
        print(f"[check] cells with accumulation >= 1000 = {100*big:.2f}% of valid "
              f"(a healthy D8 network is a few percent; a value above ~20% means the routing "
              f"is running over a flat plateau rather than over the terrain)")

    slope_rad = slope_radians_from(z, cs)
    eps = 1e-6
    a_spec = acc * cs
    tanb = np.tan(slope_rad)
    twi = np.log(np.maximum(a_spec, eps) / np.maximum(tanb, eps))
    twi[~valid] = np.nan
    const = float(np.mean(np.abs(twi[valid & np.isfinite(twi)] - np.log(cs / eps)) < 1e-3))
    if verbose:
        print(f"[info] TWI computed (acc max={np.nanmax(acc):.0f} cells; "
              f"TWI==ln(cs/eps) spike = {100*const:.3f}% of valid, want <<1%)")
    return twi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--var", default="Band1")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dx", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cs = float(args.dx)

    ds = xr.open_dataset(args.input, mask_and_scale=False)
    var = args.var if args.var in ds.data_vars else list(ds.data_vars)[0]
    z_da = ds[var]

    nodata = None
    for src in (z_da.attrs, z_da.encoding):
        for k in ("_FillValue", "missing_value", "nodata"):
            if k in src:
                try:
                    nodata = float(src[k]); break
                except Exception:
                    pass
        if nodata is not None:
            break

    ydim = z_da.dims[0]
    yv = np.asarray(ds[ydim].values, dtype=np.float64)
    y_ascending = bool(yv[1] > yv[0])

    z = np.array(z_da.values, dtype=np.float64)
    valid = np.isfinite(z) & (np.abs(z) < 1e30)
    if nodata is not None and np.isfinite(nodata):
        valid &= (z != nodata)
    z[~valid] = np.nan
    n_valid = int(valid.sum())
    zmin = float(np.nanmin(z)); zmax = float(np.nanmax(z))
    print(f"[info] DEM {z.shape}  valid={n_valid:,} ({100.0*n_valid/z.size:.1f}%)  "
          f"nodata={nodata}  cell={cs} m")
    print(f"[info] y axis is {'ascending (row 0 = south)' if y_ascending else 'descending (row 0 = north)'}; "
          f"elevation range {zmin:.2f} .. {zmax:.2f} m")
    if RD_NODATA >= zmin:
        raise SystemExit(f"[fatal] RD_NODATA={RD_NODATA} is not below the minimum elevation {zmin}")

    slope_deg, aspect_sin, aspect_cos = horn_attrs(z, cs, y_ascending)
    print(f"[info] slope zero-fraction over valid = "
          f"{100.0*np.mean(slope_deg[valid & np.isfinite(slope_deg)] == 0):.3f}%  "
          f"(median={np.nanmedian(slope_deg[valid]):.2f} deg)")

    twi = None
    if _HAS_RICHDEM:
        twi = compute_twi(z, valid, cs, read_geotransform(ds, var, cs))
    else:
        print("[warn] richdem not available -> TWI NOT written")

    coords = {d: ds[d] for d in z_da.dims if d in ds.coords}
    dims = z_da.dims

    def write(arr, name, fname):
        da = xr.DataArray(np.asarray(arr, dtype=np.float32), coords=coords, dims=dims, name=name)
        if _HAS_RIO:
            try:
                da.rio.write_crs("EPSG:2193", inplace=True)
            except Exception:
                pass
        da.encoding["_FillValue"] = float(FILL_SENTINEL)
        da.to_netcdf(os.path.join(args.outdir, fname))
        v = np.asarray(arr, dtype=np.float64); v = v[np.isfinite(v)]
        print(f"[write] {fname:16s} min/max = {v.min():.4f} / {v.max():.4f}")

    write(slope_deg, "slope_deg", "Slope_Deg.nc")
    write(aspect_sin, "aspect_sin", "Aspect_SIN.nc")
    write(aspect_cos, "aspect_cos", "Aspect_COS.nc")
    if twi is not None:
        write(twi, "TWI", "TWI.nc")

    ds.close()
    print("[done] outputs in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
