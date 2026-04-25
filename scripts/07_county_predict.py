"""
Fruit Fly Risk — County-level risk predictions.

Why county and not state:
    Saying "California is high risk in October" tells PPQ what they already know.
    The actionable question is *which counties* — Los Angeles vs. Imperial vs. San
    Diego — and that is where the data has more to say than the state aggregate.

Approach:
    The Poisson GLM was fitted at state level because that is where the validation
    labels exist (APHIS PDR records are state-tagged). We re-apply the same fitted
    coefficients to *county-level* features here. Two pieces:

    (1) Pathway features per county. Each US airport is mapped to its county via a
        spatial join on lat/lon. For each (county, month, species), sum infested
        passenger / freight volume across airports inside the county.

    (2) Network features (state_origin_entropy, state_inf_share_sp, state_n_countries)
        are inherited from the parent state — a county does not have its own origin
        mix distinct from its state's airport network.

    Then raw_mu_county = exp(Xβ) using the fitted coefficients, and we re-scale
    within each (state, month, species) so county μ's sum to the state μ. This
    preserves the calibrated state totals while distributing intelligently across
    counties using their own arrival volumes.

    Combined risk:
        county_combined = county_mu_pathway × climate_fraction
    where climate_fraction is the longest-run-of-favorable-months / 12 from
    06_climate_suitability.py. Counties with no airport get pathway=0 but may
    still have non-trivial climate scores (relevant for inland establishment).

Outputs:
    data/processed/airport_county_map.parquet
    data/processed/county_pathway_features.parquet
    data/processed/county_predictions.parquet         per (fips, month, species):
                                                       mu_pathway, climate_frac, combined,
                                                       inf_pass, state, county_name

Usage:
    python scripts/07_county_predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FIT_YEAR = 2025
SPECIES = ["capitata", "dorsalis", "ludens", "suspensa", "zonata", "cerasi"]

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


def map_airports_to_counties() -> pd.DataFrame:
    """Spatial join US airports → counties. Returns (iata_code, fips, state, county_name)."""
    a = pd.read_csv(RAW / "geo" / "airports.csv", low_memory=False)
    us = a[(a["iso_country"] == "US") & a["iata_code"].notna()
           & a["latitude_deg"].notna() & a["longitude_deg"].notna()].copy()
    us = us.drop_duplicates("iata_code")

    pts = gpd.GeoDataFrame(
        us[["iata_code"]],
        geometry=gpd.points_from_xy(us["longitude_deg"], us["latitude_deg"]),
        crs="EPSG:4326",
    )
    counties = gpd.read_file(RAW / "geo" / "us_counties.geojson")
    counties["fips"] = counties["STATE"] + counties["COUNTY"]

    joined = gpd.sjoin(pts, counties[["fips", "STATE", "NAME", "geometry"]],
                       how="inner", predicate="within")
    joined["state"] = joined["STATE"].map(FIPS_TO_STATE)
    out = joined[["iata_code", "fips", "state", "NAME"]].rename(columns={"NAME": "county_name"})
    return out


def build_county_features(amap: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate per-airport infested volumes to (county, month, species)."""
    rt = pd.read_parquet(PROCESSED / "risk_table.parquet")
    rt = rt[rt["year"] == year].merge(
        amap.rename(columns={"iata_code": "dest_us_port"}),
        on="dest_us_port", how="inner",
    )
    rows = []
    for sp in SPECIES:
        pcol = f"present_{sp}"
        sub = rt[["fips", "state", "county_name", "year", "month", "passengers",
                  "freight_kg", "host_kg_total", pcol]].copy()
        sub["inf_pass"]    = np.where(sub[pcol] == 1, sub["passengers"],    0.0)
        sub["inf_freight"] = np.where(sub[pcol] == 1, sub["freight_kg"],    0.0)
        sub["inf_host"]    = np.where(sub[pcol] == 1, sub["host_kg_total"], 0.0)
        g = sub.groupby(["fips", "state", "county_name", "year", "month"], as_index=False).agg(
            inf_pass=("inf_pass", "sum"),
            inf_freight=("inf_freight", "sum"),
            inf_host=("inf_host", "sum"))
        g["species"] = sp
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def predict_county_mu(features: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted GLM coefficients at the county level (Fourier seasonality, log
    volumes, species dummies, network features inherited from state).
    Then scale within (state, month, species) so county μ's sum to state μ."""
    coefs = pd.read_csv(PROCESSED / "model_coefficients.csv")
    beta = dict(zip(coefs["term"], coefs["coef"]))
    state_net = pd.read_parquet(PROCESSED / "state_network_features.parquet")
    state_net = state_net[state_net["year"] == FIT_YEAR].drop(columns=["year"])

    df = features.copy()
    df["log_pass"]    = np.log1p(df["inf_pass"])
    df["log_freight"] = np.log1p(df["inf_freight"])
    df["log_host"]    = np.log1p(df["inf_host"])

    angle = 2 * np.pi * df["month"] / 12.0
    df["sin_m1"] = np.sin(angle)
    df["cos_m1"] = np.cos(angle)
    df["sin_m2"] = np.sin(2 * angle)
    df["cos_m2"] = np.cos(2 * angle)

    df = df.merge(state_net, on=["state", "month"], how="left")
    # Species-specific infested share: pick the column matching this row's species.
    sp_share_cols = [c for c in df.columns if c.startswith("state_inf_share_")]
    df["state_inf_share_sp"] = 0.0
    for sp in SPECIES:
        col = f"state_inf_share_{sp}"
        if col in df.columns:
            mask = df["species"] == sp
            df.loc[mask, "state_inf_share_sp"] = df.loc[mask, col].fillna(0.0)
    df["log_state_n_countries"] = np.log1p(df["state_n_countries"].fillna(0))
    for c in ["state_origin_entropy", "state_inf_share_sp", "log_state_n_countries"]:
        df[c] = df[c].fillna(0.0)

    # Species fixed effects: capitata is the reference (drop_first=True at fit time).
    species_terms = {sp: beta.get(f"species_{sp}", 0.0) for sp in SPECIES}
    species_terms["capitata"] = 0.0
    df["species_term"] = df["species"].map(species_terms)

    eta = (
        beta.get("const", 0)
        + beta.get("log_pass", 0)              * df["log_pass"]
        + beta.get("log_freight", 0)           * df["log_freight"]
        + beta.get("log_host", 0)              * df["log_host"]
        + beta.get("sin_m1", 0)                * df["sin_m1"]
        + beta.get("cos_m1", 0)                * df["cos_m1"]
        + beta.get("sin_m2", 0)                * df["sin_m2"]
        + beta.get("cos_m2", 0)                * df["cos_m2"]
        + beta.get("log_state_n_countries", 0) * df["log_state_n_countries"]
        + beta.get("state_origin_entropy", 0)  * df["state_origin_entropy"]
        + beta.get("state_inf_share_sp", 0)    * df["state_inf_share_sp"]
        + df["species_term"]
    )
    df["raw_mu"] = np.exp(eta)

    # Re-scale within (state, month, species) to match the state-level μ.
    state_pred = pd.read_parquet(PROCESSED / "risk_predictions.parquet")
    state_target = state_pred.rename(columns={"mu": "state_mu"})[["state", "month", "species", "state_mu"]]
    df = df.merge(state_target, on=["state", "month", "species"], how="left")

    sums = df.groupby(["state", "month", "species"])["raw_mu"].transform("sum")
    df["mu_pathway"] = np.where(sums > 0, df["raw_mu"] * df["state_mu"] / sums, 0.0)

    # Propagate state-level Poisson 80% credible intervals down to county level by
    # the same scaling factor. The interval is wide because of the small validation
    # sample, but the relative ratios between counties within a state are preserved.
    ci = state_pred.rename(columns={"mu_lo80": "state_mu_lo80", "mu_hi80": "state_mu_hi80"})[
        ["state", "month", "species", "state_mu_lo80", "state_mu_hi80"]
    ]
    df = df.merge(ci, on=["state", "month", "species"], how="left")
    state_mu_safe = df["state_mu"].replace(0, np.nan)
    lo_factor = (df["state_mu_lo80"] / state_mu_safe).fillna(0.0)
    hi_factor = (df["state_mu_hi80"] / state_mu_safe).fillna(0.0)
    df["mu_pathway_lo80"] = df["mu_pathway"] * lo_factor
    df["mu_pathway_hi80"] = df["mu_pathway"] * hi_factor

    return df[["fips", "state", "county_name", "month", "species", "year",
               "inf_pass", "inf_freight", "inf_host",
               "raw_mu", "state_mu", "mu_pathway",
               "mu_pathway_lo80", "mu_pathway_hi80"]]


def attach_climate_and_combine(predictions: pd.DataFrame) -> pd.DataFrame:
    """Multiply pathway μ by climate suitability fraction to get combined county risk."""
    cs = pd.read_parquet(PROCESSED / "climate_suitability_by_county.parquet")
    cs["frac_year_favorable"] = cs["long_run_mean"].fillna(0) / 12.0
    cs = cs[["fips", "species", "long_run_mean", "frac_year_favorable"]].copy()
    cs["fips"] = cs["fips"].astype(str).str.zfill(5)
    predictions["fips"] = predictions["fips"].astype(str).str.zfill(5)
    out = predictions.merge(cs, on=["fips", "species"], how="left")
    out["frac_year_favorable"] = out["frac_year_favorable"].fillna(0.0)
    out["long_run_mean"]       = out["long_run_mean"].fillna(0.0)
    out["combined"] = out["mu_pathway"] * out["frac_year_favorable"]
    if "mu_pathway_lo80" in out.columns:
        out["combined_lo80"] = out["mu_pathway_lo80"] * out["frac_year_favorable"]
        out["combined_hi80"] = out["mu_pathway_hi80"] * out["frac_year_favorable"]
    return out


def report(df: pd.DataFrame) -> None:
    print("\n=== County-level risk predictions ===")
    print(f"Rows:                 {len(df):,}")
    print(f"Counties with arrivals: {(df['mu_pathway'] > 0).groupby(df['fips']).any().sum()}")
    print(f"Counties with combined > 0: {(df['combined'] > 0).groupby(df['fips']).any().sum()}")

    for sp in SPECIES:
        peak = df[df["species"] == sp].copy()
        # Annual aggregates per county
        annual = peak.groupby(["fips", "state", "county_name"], as_index=False).agg(
            mu_year=("mu_pathway", "sum"),
            combined_year=("combined", "sum"),
            climate_frac=("frac_year_favorable", "first"),
        )
        print(f"\nTop 15 counties by COMBINED risk for {sp}:")
        top = annual.nlargest(15, "combined_year")
        for _, r in top.iterrows():
            bar = "█" * int(min(r["combined_year"] * 30, 40))
            print(f"  {r['state']}  {r['county_name'][:24]:24s}  "
                  f"μ={r['mu_year']:5.2f}  clim={r['climate_frac']:.2f}  "
                  f"combined={r['combined_year']:5.2f}  {bar}")


def predict_county_history(amap: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Compute annual county-level predictions for past years using the fitted
    coefficients applied directly (no state-rescaling — we don't have fitted
    state μ's for historical years). Useful for YoY sparklines: the shape of the
    trend is meaningful even if absolute counts are uncalibrated."""
    coefs = pd.read_csv(PROCESSED / "model_coefficients.csv")
    beta = dict(zip(coefs["term"], coefs["coef"]))
    state_net_all = pd.read_parquet(PROCESSED / "state_network_features.parquet")
    cs = pd.read_parquet(PROCESSED / "climate_suitability_by_county.parquet")
    cs["frac_year_favorable"] = cs["long_run_mean"].fillna(0) / 12.0
    cs = cs[["fips", "species", "frac_year_favorable"]].copy()
    cs["fips"] = cs["fips"].astype(str).str.zfill(5)

    species_terms = {sp: beta.get(f"species_{sp}", 0.0) for sp in SPECIES}
    species_terms["capitata"] = 0.0

    out_rows = []
    for year in years:
        feats = build_county_features(amap, year)
        if feats.empty:
            continue

        df = feats.copy()
        df["log_pass"]    = np.log1p(df["inf_pass"])
        df["log_freight"] = np.log1p(df["inf_freight"])
        df["log_host"]    = np.log1p(df["inf_host"])
        angle = 2 * np.pi * df["month"] / 12.0
        df["sin_m1"] = np.sin(angle); df["cos_m1"] = np.cos(angle)
        df["sin_m2"] = np.sin(2 * angle); df["cos_m2"] = np.cos(2 * angle)

        sn_yr = state_net_all[state_net_all["year"] == year].drop(columns=["year"])
        df = df.merge(sn_yr, on=["state", "month"], how="left")
        df["state_inf_share_sp"] = 0.0
        for sp in SPECIES:
            col = f"state_inf_share_{sp}"
            if col in df.columns:
                mask = df["species"] == sp
                df.loc[mask, "state_inf_share_sp"] = df.loc[mask, col].fillna(0.0)
        df["log_state_n_countries"] = np.log1p(df.get("state_n_countries", 0).fillna(0))
        for c in ["state_origin_entropy", "state_inf_share_sp", "log_state_n_countries"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df["species_term"] = df["species"].map(species_terms)

        eta = (
            beta.get("const", 0)
            + beta.get("log_pass", 0)              * df["log_pass"]
            + beta.get("log_freight", 0)           * df["log_freight"]
            + beta.get("log_host", 0)              * df["log_host"]
            + beta.get("sin_m1", 0)                * df["sin_m1"]
            + beta.get("cos_m1", 0)                * df["cos_m1"]
            + beta.get("sin_m2", 0)                * df["sin_m2"]
            + beta.get("cos_m2", 0)                * df["cos_m2"]
            + beta.get("log_state_n_countries", 0) * df["log_state_n_countries"]
            + beta.get("state_origin_entropy", 0)  * df["state_origin_entropy"]
            + beta.get("state_inf_share_sp", 0)    * df["state_inf_share_sp"]
            + df["species_term"]
        )
        df["raw_mu"] = np.exp(eta)
        df["fips"] = df["fips"].astype(str).str.zfill(5)
        df = df.merge(cs, on=["fips", "species"], how="left")
        df["frac_year_favorable"] = df["frac_year_favorable"].fillna(0)
        df["combined"] = df["raw_mu"] * df["frac_year_favorable"]
        ann = df.groupby(["fips", "state", "county_name", "species"], as_index=False).agg(
            annual_pathway=("raw_mu", "sum"),
            annual_combined=("combined", "sum"))
        ann["year"] = year
        out_rows.append(ann)

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def main() -> int:
    print("Mapping US airports to counties (spatial join)…")
    amap = map_airports_to_counties()
    print(f"  airports mapped: {len(amap):,}")
    amap.to_parquet(PROCESSED / "airport_county_map.parquet", index=False, compression="zstd")

    print("Building county-level pathway features…")
    feats = build_county_features(amap, FIT_YEAR)
    print(f"  feature rows: {len(feats):,}  ({feats['fips'].nunique()} counties × 12 months × 3 species)")
    feats.to_parquet(PROCESSED / "county_pathway_features.parquet", index=False, compression="zstd")

    print("Applying fitted GLM coefficients per county and rescaling…")
    preds = predict_county_mu(feats)

    print("Attaching climate suitability and computing combined risk…")
    out = attach_climate_and_combine(preds)
    out.to_parquet(PROCESSED / "county_predictions.parquet", index=False, compression="zstd")

    print("\nComputing 5-year YoY history (raw, un-rescaled — shape only)…")
    history_years = [FIT_YEAR - i for i in range(5, 0, -1)] + [FIT_YEAR]
    hist = predict_county_history(amap, history_years)
    if not hist.empty:
        hist.to_parquet(PROCESSED / "county_predictions_history.parquet",
                        index=False, compression="zstd")
        print(f"  history rows: {len(hist):,}  ({hist['year'].nunique()} years × {hist['fips'].nunique()} counties × {hist['species'].nunique()} species)")

    report(out)
    print(f"\nWrote data/processed/airport_county_map.parquet      ({len(amap):,} rows)")
    print(f"Wrote data/processed/county_pathway_features.parquet  ({len(feats):,} rows)")
    print(f"Wrote data/processed/county_predictions.parquet      ({len(out):,} rows)")
    print(f"Wrote data/processed/county_predictions_history.parquet  ({len(hist) if not hist.empty else 0:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
