"""
Fruit Fly Risk — Climate suitability per US state and county.

For each target species, compute the LONGEST RUN OF CONSECUTIVE FAVORABLE MONTHS
(cyclic over Jan-Dec) per pixel of the WorldClim 10-arcmin grid. A month is
"favorable" when monthly mean temperature is in the species' tolerance band AND
monthly precipitation exceeds the species minimum.

Why "longest cyclic run" instead of "fraction of year favorable":
    Establishment requires consecutive generations. A county with 6 favorable
    months scattered across the year cannot sustain a population the way a
    county with 6 *consecutive* favorable months can. The cyclic max captures
    this.

Climate envelopes (literature-grounded approximations):
    Ceratitis capitata     tavg 16-32 C, prec ≥ 20 mm
    Bactrocera dorsalis    tavg 13-32 C, prec ≥ 50 mm   (broader cool tolerance)
    Anastrepha ludens      tavg 14-30 C, prec ≥ 30 mm

Aggregation:
    Per-pixel longest-run rasters are area-aggregated to US counties via
    rasterstats, then counties are averaged to states (simple mean — the
    distribution of 'favorability' across counties is more informative than
    a single state value, but for the model integration we want one number).

Outputs:
    data/processed/climate_grid_<species>.tif        per-species CONUS raster (months 0-12)
    data/processed/climate_suitability_by_county.parquet
    data/processed/climate_suitability_by_state.parquet

Usage:
    python scripts/06_climate_suitability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Species climate envelopes (tavg in deg C, prec in mm/month). Cherry fly (cerasi)
# is a temperate species — fundamentally different ecology from the tropical/
# subtropical fruit flies, hence the much lower upper temperature bound.
ENVELOPES = {
    "capitata": {"tavg_min": 16, "tavg_max": 32, "prec_min": 20},
    "dorsalis": {"tavg_min": 13, "tavg_max": 32, "prec_min": 50},
    "ludens":   {"tavg_min": 14, "tavg_max": 30, "prec_min": 30},
    "suspensa": {"tavg_min": 16, "tavg_max": 32, "prec_min": 30},  # Caribbean — humid subtropical
    "zonata":   {"tavg_min": 14, "tavg_max": 32, "prec_min": 40},  # tropical Asia/Africa
    "cerasi":   {"tavg_min":  8, "tavg_max": 25, "prec_min": 30},  # temperate Eurasian cherry fly
}

# Contiguous US bounding box (slightly wider for raster border safety).
CONUS = {"left": -125.5, "right": -66.0, "bottom": 24.0, "top": 50.0}

# Standard FIPS → 2-letter state map for the contiguous US plus AK/HI (the
# inbound flight ports include both — we keep them in the model panel).
FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def load_monthly_stack(var: str) -> tuple[np.ndarray, rasterio.Affine]:
    """Load 12 monthly rasters of `var` ('tavg' or 'prec'), CONUS-cropped, into (12, H, W)."""
    files = sorted((RAW / "climate").glob(f"wc2.1_10m_{var}_*.tif"))
    if len(files) != 12:
        sys.exit(f"Expected 12 {var} rasters, found {len(files)}")

    with rasterio.open(files[0]) as src:
        win = from_bounds(CONUS["left"], CONUS["bottom"], CONUS["right"], CONUS["top"],
                          transform=src.transform)
        win = win.round_lengths().round_offsets()
        arr0 = src.read(1, window=win, masked=True).astype(np.float32)
        out = np.zeros((12, arr0.shape[0], arr0.shape[1]), dtype=np.float32)
        out[0] = arr0.filled(np.nan)
        new_transform = src.window_transform(win)

    for i, f in enumerate(files[1:], start=1):
        with rasterio.open(f) as src:
            arr = src.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
            out[i] = arr
    return out, new_transform


def longest_consec_true(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """Vectorized longest run of True along `axis`."""
    m = mask.astype(np.int16)
    cs = np.cumsum(m, axis=axis)
    reset_marker = np.where(mask, 0, cs)
    reset_at = np.maximum.accumulate(reset_marker, axis=axis)
    run_lengths = (cs - reset_at) * m
    return run_lengths.max(axis=axis)


def longest_cyclic_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True considering Jan-Dec wraparound. mask shape (12, H, W)."""
    doubled = np.concatenate([mask, mask], axis=0)  # (24, H, W)
    return np.minimum(longest_consec_true(doubled, axis=0), 12).astype(np.uint8)


def write_geotiff(path: Path, data: np.ndarray, transform: rasterio.Affine) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width":  data.shape[1],
        "count":  1,
        "dtype":  data.dtype,
        "crs":    "EPSG:4326",
        "transform": transform,
        "compress":  "deflate",
        "nodata": 255 if data.dtype == np.uint8 else None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def aggregate_to_counties(grid_paths: dict[str, Path]) -> pd.DataFrame:
    """For each species' raster, compute per-county area-weighted mean of the longest-run score."""
    counties_path = RAW / "geo" / "us_counties.geojson"
    rows: list[dict] = []
    for sp, raster in grid_paths.items():
        stats = zonal_stats(
            str(counties_path), str(raster),
            stats=["mean", "max", "count"],
            geojson_out=True,
            all_touched=False,
            nodata=255,
        )
        for feat in stats:
            props = feat["properties"]
            fips_state  = props.get("STATE", "")
            fips_county = props.get("COUNTY", "")
            if not fips_state or not fips_county:
                continue
            rows.append({
                "fips_state":  fips_state,
                "fips_county": fips_county,
                "fips":        f"{fips_state}{fips_county}",
                "county_name": props.get("NAME"),
                "state":       FIPS_TO_STATE.get(fips_state, fips_state),
                "species":     sp,
                "long_run_mean": props.get("mean"),
                "long_run_max":  props.get("max"),
                "pixel_count":   props.get("count", 0),
            })
    return pd.DataFrame(rows)


def aggregate_to_states(by_county: pd.DataFrame) -> pd.DataFrame:
    """County → state simple mean (per species)."""
    out = (by_county.dropna(subset=["long_run_mean"])
                    .groupby(["state", "species"], as_index=False)
                    .agg(long_run_mean=("long_run_mean", "mean"),
                         long_run_max=("long_run_max", "max"),
                         n_counties=("fips", "nunique")))
    out["frac_year_favorable"] = out["long_run_mean"] / 12.0
    return out


def main() -> int:
    print("Loading WorldClim tavg and prec stacks (CONUS)…")
    tavg, transform = load_monthly_stack("tavg")
    prec, _         = load_monthly_stack("prec")
    print(f"  stack shape: {tavg.shape}  (12, H, W)")

    grid_paths: dict[str, Path] = {}
    print("\nComputing per-species climate envelopes…")
    for sp, env in ENVELOPES.items():
        favorable = (
            (tavg >= env["tavg_min"]) &
            (tavg <= env["tavg_max"]) &
            (prec >= env["prec_min"])
        )
        # NaN propagation — pixels with any NaN month are unusable.
        any_nan = np.isnan(tavg).any(axis=0) | np.isnan(prec).any(axis=0)
        long_run = longest_cyclic_run(favorable)
        long_run[any_nan] = 255  # nodata sentinel (uint8)

        grid_path = PROCESSED / f"climate_grid_{sp}.tif"
        write_geotiff(grid_path, long_run, transform)
        grid_paths[sp] = grid_path

        valid = long_run[long_run != 255]
        n_pix = valid.size
        n_4_or_more = int((valid >= 4).sum())
        n_6_or_more = int((valid >= 6).sum())
        n_full_year = int((valid == 12).sum())
        print(f"  {sp}: {n_pix:,} CONUS pixels  |  ≥4mo {n_4_or_more:,} ({n_4_or_more/n_pix*100:.1f}%)"
              f"  |  ≥6mo {n_6_or_more:,} ({n_6_or_more/n_pix*100:.1f}%)"
              f"  |  full-year {n_full_year:,} ({n_full_year/n_pix*100:.1f}%)")

    print("\nAggregating to US counties (zonal stats over us_counties.geojson)…")
    by_county = aggregate_to_counties(grid_paths)
    print(f"  county rows (county × species): {len(by_county):,}")

    by_state = aggregate_to_states(by_county)
    by_county.to_parquet(PROCESSED / "climate_suitability_by_county.parquet",
                         index=False, compression="zstd")
    by_state.to_parquet(PROCESSED / "climate_suitability_by_state.parquet",
                        index=False, compression="zstd")

    print("\nTop 10 states by climate suitability (longest cyclic run, mean across counties):")
    for sp in ENVELOPES:
        top = by_state[by_state["species"] == sp].nlargest(10, "long_run_mean")
        print(f"\n  {sp}:")
        for _, r in top.iterrows():
            bar = "█" * int(r["long_run_mean"])
            print(f"    {r['state']}  {r['long_run_mean']:5.2f} mo  {bar}")

    print(f"\nWrote data/processed/climate_grid_{{capitata,dorsalis,ludens}}.tif")
    print(f"Wrote data/processed/climate_suitability_by_county.parquet  ({len(by_county):,} rows)")
    print(f"Wrote data/processed/climate_suitability_by_state.parquet  ({len(by_state):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
