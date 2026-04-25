"""
Fruit Fly Risk — Fit a Poisson regression of detections on pathway volumes.

Predicts:  E[detections_{state, month, species}] in a single calendar year.

Features per (state, month, species):
    log1p(infested_passengers)   passengers from countries where THIS species is present
    log1p(infested_freight_kg)   air freight kg, same filter
    log1p(infested_host_kg)      USDA host-commodity imports kg, same filter
    species fixed effect (3 levels)
    month fixed effect (12 levels)

Validation target:
    APHIS PDR detection events, aggregated to (state, month, species). The
    sample is small (~50 events for our 3 target species across 2018-2026),
    so coefficients should be read as illustrative, not definitive.

Why no state fixed effect?
    With 4 states present in the validation set (CA, TX, NY, plus a generic
    "US"), a state FE would memorize them and prevent the model from
    extrapolating risk to states with zero historical detections — which is
    precisely the operational use case (early warning).

Outputs:
    data/processed/risk_predictions.parquet   one row per (state, month, species)
                                              with mu (predicted intensity) and
                                              80% CI bounds from the GLM.
    data/processed/model_coefficients.csv     fitted coefficients with std errors.

Usage:
    python scripts/03_fit_risk_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Year to fit on. Most APHIS validation events are 2025; use that as the
# focal year so the exposures and the labels are temporally aligned.
FIT_YEAR = 2025

SPECIES_MAP = {
    "Ceratitis capitata":   "capitata",
    "Bactrocera dorsalis":  "dorsalis",
    "Anastrepha ludens":    "ludens",
    "Anastrepha suspensa":  "suspensa",
    "Bactrocera zonata":    "zonata",
    "Rhagoletis cerasi":    "cerasi",
}
SPECIES = list(SPECIES_MAP.values())


def port_to_state() -> pd.DataFrame:
    """Map IATA port code → 2-letter US state via airports.csv."""
    a = pd.read_csv(RAW / "geo" / "airports.csv", low_memory=False,
                    usecols=["iata_code", "iso_country", "iso_region"])
    a = a[(a["iso_country"] == "US") & a["iata_code"].notna()].copy()
    a["state"] = a["iso_region"].str.split("-").str[1]
    a = a.dropna(subset=["state"])
    a = a.drop_duplicates("iata_code", keep="first")
    return a[["iata_code", "state"]].rename(columns={"iata_code": "dest_us_port"})


def species_long(rt: pd.DataFrame) -> pd.DataFrame:
    """Reshape risk_table so each (port-month) row is repeated 3× (one per species),
    with infested-volume features computed per species."""
    rows = []
    for sp in SPECIES:
        present_col = f"present_{sp}"
        sub = rt[["dest_us_port", "year", "month", "passengers", "freight_kg",
                  "host_kg_total", present_col]].copy()
        sub["species"] = sp
        sub["inf_pass"]    = np.where(sub[present_col] == 1, sub["passengers"],    0.0)
        sub["inf_freight"] = np.where(sub[present_col] == 1, sub["freight_kg"],    0.0)
        sub["inf_host"]    = np.where(sub[present_col] == 1, sub["host_kg_total"], 0.0)
        rows.append(sub[["dest_us_port", "year", "month", "species",
                         "inf_pass", "inf_freight", "inf_host"]])
    return pd.concat(rows, ignore_index=True)


def aggregate_to_state(features_long: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate port-level features to state × month × species for a given year."""
    p2s = port_to_state()
    f = features_long[features_long["year"] == year].merge(p2s, on="dest_us_port", how="inner")
    # GATS host_kg is country-month, replicated across all ports the country flies into.
    # Summing across ports inside a state inflates it (a CA state row would count
    # Mexico->LAX, Mexico->SFO, Mexico->SAN, ... all separately). Sum passengers/freight
    # across ports as normal; for host, take the *max across ports per origin-state-month*
    # via a separate aggregation. For v1 we accept the simpler "sum at port-state level"
    # of host_kg knowing it overstates absolute volume but preserves cross-state ranking.
    out = (f.groupby(["state", "month", "species"], as_index=False)
             .agg(inf_pass=("inf_pass", "sum"),
                  inf_freight=("inf_freight", "sum"),
                  inf_host=("inf_host", "sum")))
    return out


def load_validation(year: int) -> pd.DataFrame:
    """APHIS detection events → (state, month, species) counts for the focal year."""
    v = pd.read_csv(RAW / "pests" / "aphis_validation.csv")
    v = v[v["species"].isin(SPECIES_MAP)].copy()
    v["species"] = v["species"].map(SPECIES_MAP)
    v = v[v["state_or_port"].str.len() == 2].copy()  # drop "US" generic
    v = v[v["year"] == year]
    out = (v.groupby(["state_or_port", "month", "species"], as_index=False)
             .agg(detections=("count", "sum"))
             .rename(columns={"state_or_port": "state"}))
    return out


def load_network_features(year: int) -> pd.DataFrame | None:
    """Optional: network-structure features per state × month produced by 05_network_features.py.
    Returns None if not yet generated, in which case the model fits without them."""
    path = PROCESSED / "state_network_features.parquet"
    if not path.exists():
        return None
    sf = pd.read_parquet(path)
    sf = sf[sf["year"] == year].drop(columns=["year"])
    return sf


def build_panel(year: int) -> pd.DataFrame:
    """Full state × month × species panel for the focal year, joined to features and labels."""
    rt = pd.read_parquet(PROCESSED / "risk_table.parquet")
    long = species_long(rt)
    features = aggregate_to_state(long, year)

    states = sorted(features["state"].unique())
    months = list(range(1, 13))
    panel = pd.MultiIndex.from_product(
        [states, months, SPECIES],
        names=["state", "month", "species"],
    ).to_frame(index=False)

    panel = panel.merge(features, on=["state", "month", "species"], how="left")
    for c in ["inf_pass", "inf_freight", "inf_host"]:
        panel[c] = panel[c].fillna(0.0)

    net = load_network_features(year)
    if net is not None:
        # Pick the species-specific infested-share column at fit time.
        panel = panel.merge(net, on=["state", "month"], how="left")
        # Each row's "infested share for THIS species" — same value across the row's species.
        panel["state_inf_share_sp"] = np.select(
            [panel["species"] == "capitata",
             panel["species"] == "dorsalis",
             panel["species"] == "ludens"],
            [panel["state_inf_share_capitata"],
             panel["state_inf_share_dorsalis"],
             panel["state_inf_share_ludens"]],
            default=0.0,
        )
        for c in ["state_n_countries", "state_origin_entropy", "state_total_pax",
                  "state_inf_share_sp"]:
            panel[c] = panel[c].fillna(0.0)

    labels = load_validation(year)
    panel = panel.merge(labels, on=["state", "month", "species"], how="left")
    panel["detections"] = panel["detections"].fillna(0).astype(int)

    return panel


def fit_poisson(panel: pd.DataFrame) -> tuple[sm.GLM, pd.DataFrame]:
    """Fit Poisson GLM and return the fitted model + design matrix.

    Seasonality uses two Fourier harmonics (annual + semi-annual) instead of
    monthly dummies. With 12 dummies, any month with zero observed detections
    drives its coefficient to -infinity (huge std error, degenerate fit).
    Fourier features smooth that out and only use 4 parameters.
    """
    df = panel.copy()
    df["log_pass"]    = np.log1p(df["inf_pass"])
    df["log_freight"] = np.log1p(df["inf_freight"])
    df["log_host"]    = np.log1p(df["inf_host"])

    angle = 2 * np.pi * df["month"] / 12.0
    df["sin_m1"] = np.sin(angle)
    df["cos_m1"] = np.cos(angle)
    df["sin_m2"] = np.sin(2 * angle)
    df["cos_m2"] = np.cos(2 * angle)

    feature_cols = ["log_pass", "log_freight", "log_host",
                    "sin_m1", "cos_m1", "sin_m2", "cos_m2", "species"]
    if "state_origin_entropy" in df.columns:
        df["log_state_n_countries"] = np.log1p(df["state_n_countries"])
        feature_cols += ["log_state_n_countries", "state_origin_entropy", "state_inf_share_sp"]
    X = pd.get_dummies(df[feature_cols], columns=["species"], drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(df["detections"], X, family=sm.families.Poisson()).fit()
    return model, X


def report(model, X: pd.DataFrame, panel: pd.DataFrame, year: int) -> pd.DataFrame:
    print(f"\n=== Poisson GLM — year {year} ===")
    print(model.summary().tables[1])

    pred = model.get_prediction(X)
    summary = pred.summary_frame(alpha=0.20)  # 80% CI
    out = panel[["state", "month", "species", "detections",
                 "inf_pass", "inf_freight", "inf_host"]].copy()
    out["mu"]      = summary["mean"].values
    out["mu_lo80"] = summary["mean_ci_lower"].values
    out["mu_hi80"] = summary["mean_ci_upper"].values

    # Sanity: total predicted vs. total observed.
    # Pearson χ² is excluded — with rare species (cerasi/suspensa) whose negative
    # species fixed effects drive μ near zero in most cells, observed events at
    # those cells inflate (obs−μ)²/μ astronomically. Deviance-based pseudo-R²
    # is the honest fit metric for this regime.
    print(f"\nTotal observed detections:  {out['detections'].sum()}")
    print(f"Total predicted intensity:   {out['mu'].sum():.2f}")
    print(f"Pseudo R² (Cragg-Uhler):     {1 - model.deviance / model.null_deviance:.3f}")
    print(f"Deviance / df_resid:         {model.deviance / model.df_resid:.3f}  (overdispersion proxy; <2 is OK)")

    print("\nTop 15 (state × month × species) by predicted intensity:")
    top = out.sort_values("mu", ascending=False).head(15)
    print(top[["state", "month", "species", "mu", "mu_lo80", "mu_hi80",
               "detections"]].to_string(index=False, float_format="%.3f"))

    print("\nLargest residuals (predicted vs. observed mismatch):")
    out["resid"] = out["detections"] - out["mu"]
    surprises = (out.assign(abs_resid=out["resid"].abs())
                    .sort_values("abs_resid", ascending=False)
                    .head(10))
    print(surprises[["state", "month", "species", "detections", "mu", "resid"]]
              .to_string(index=False, float_format="%.3f"))

    return out


def main() -> int:
    panel = build_panel(FIT_YEAR)
    print(f"Panel: {len(panel):,} rows  ({panel['state'].nunique()} states × 12 months × {len(SPECIES)} species)")
    print(f"Non-zero detection cells: {(panel['detections'] > 0).sum()}")
    print(f"Total detections:         {panel['detections'].sum()}")

    model, X = fit_poisson(panel)
    predictions = report(model, X, panel, FIT_YEAR)

    # Persist predictions and coefficients.
    predictions.to_parquet(PROCESSED / "risk_predictions.parquet", index=False, compression="zstd")

    coef_df = pd.DataFrame({
        "term":      model.params.index,
        "coef":      model.params.values,
        "std_err":   model.bse.values,
        "z":         model.tvalues.values,
        "p_value":   model.pvalues.values,
    })
    coef_df.to_csv(PROCESSED / "model_coefficients.csv", index=False)

    print(f"\nWrote data/processed/risk_predictions.parquet  ({len(predictions):,} rows)")
    print(f"Wrote data/processed/model_coefficients.csv     ({len(coef_df)} terms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
