"""
Fruit Fly Risk — Network-structure features per US port and state.

True 1-stop path attribution (e.g. Vietnam → Korea → LAX) is not possible from
T-100 International Segment data alone — it is strictly bilateral (no
foreign-to-foreign segments and no US-domestic segments). Instead, we
characterize each port's *position* in the global inbound network with three
feature families:

    diversity           number of distinct foreign origin countries served,
                        and Shannon entropy of inbound pax distribution
                        (a high-entropy port is a heterogeneous gateway —
                         systematically harder to surveille than a
                         single-route port like Mexico → Brownsville)

    infested-share      fraction of inbound pax from countries where each
                        target species is present (a species-specific
                        exposure ratio, distinct from raw infested volume)

    concentration       per-origin-country HHI of US port distribution
                        (countries that funnel all their US traffic into
                         one port behave differently from countries that
                         spread across many)

These are added at port and state levels and can be merged into the model
panel as additional predictors.

Output:
    data/processed/port_network_features.parquet
    data/processed/state_network_features.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SPECIES = ["capitata", "dorsalis", "ludens", "suspensa", "zonata", "cerasi"]


def shannon(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def hhi(p: np.ndarray) -> float:
    """Herfindahl-Hirschman concentration index (0..1)."""
    if p.sum() == 0:
        return 0.0
    s = p / p.sum()
    return float((s ** 2).sum())


def port_features(rt: pd.DataFrame) -> pd.DataFrame:
    """Per (dest_us_port, year, month) network-structure features."""
    rows = []
    grouped = rt.groupby(["dest_us_port", "year", "month"], sort=False)
    for (port, yr, mo), sub in grouped:
        p = sub["passengers"].to_numpy(dtype=float)
        rec = {
            "dest_us_port": port,
            "year": yr,
            "month": mo,
            "port_n_countries":     int((p > 0).sum()),
            "port_origin_entropy":  shannon(p),
            "port_total_pax":       float(p.sum()),
        }
        total_pax = p.sum()
        for sp in SPECIES:
            mask = sub[f"present_{sp}"] == 1
            inf_pax = float(sub.loc[mask, "passengers"].sum())
            rec[f"port_inf_share_{sp}"] = inf_pax / total_pax if total_pax > 0 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def country_concentration(rt: pd.DataFrame) -> pd.DataFrame:
    """Per (origin_country, year, month) — HHI across destination US ports."""
    grouped = rt.groupby(["origin_country", "year", "month"], sort=False)
    rows = []
    for (cc, yr, mo), sub in grouped:
        rows.append({
            "origin_country": cc,
            "year": yr,
            "month": mo,
            "origin_us_port_hhi": hhi(sub["passengers"].to_numpy(dtype=float)),
            "origin_us_port_count": int((sub["passengers"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def state_features(port_feat: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate port features to state level via airports.csv."""
    a = pd.read_csv(RAW / "geo" / "airports.csv", low_memory=False,
                    usecols=["iata_code", "iso_country", "iso_region"])
    a = a[(a["iso_country"] == "US") & a["iata_code"].notna()].copy()
    a["state"] = a["iso_region"].str.split("-").str[1]
    p2s = a.dropna(subset=["state"]).drop_duplicates("iata_code")[["iata_code", "state"]]
    p2s = p2s.rename(columns={"iata_code": "dest_us_port"})

    rt_state = rt.merge(p2s, on="dest_us_port", how="inner")
    present_cols = [f"present_{sp}" for sp in SPECIES if f"present_{sp}" in rt_state.columns]
    rows = []
    for (st, yr, mo), sub in rt_state.groupby(["state", "year", "month"], sort=False):
        # Re-aggregate by origin_country since one country may fly to multiple ports in the state.
        agg_spec = {"pax": ("passengers", "sum")}
        for col in present_cols:
            agg_spec[col] = (col, "max")
        by_country = sub.groupby("origin_country").agg(**agg_spec)
        p = by_country["pax"].to_numpy(dtype=float)
        rec = {
            "state": st,
            "year":  yr,
            "month": mo,
            "state_n_countries":    int((p > 0).sum()),
            "state_origin_entropy": shannon(p),
            "state_total_pax":      float(p.sum()),
        }
        total = p.sum()
        for sp in SPECIES:
            col = f"present_{sp}"
            if col in by_country.columns:
                inf = float(by_country.loc[by_country[col] == 1, "pax"].sum())
                rec[f"state_inf_share_{sp}"] = inf / total if total > 0 else 0.0
            else:
                rec[f"state_inf_share_{sp}"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    rt = pd.read_parquet(PROCESSED / "risk_table.parquet")
    print(f"Loaded risk table: {len(rt):,} rows")

    print("Computing port-level features…")
    port_feat = port_features(rt)
    cc_feat   = country_concentration(rt)
    print(f"  ports:    {len(port_feat):,} rows  (port × month-year)")
    print(f"  countries:{len(cc_feat):,} rows  (origin × month-year)")

    print("Computing state-level features…")
    state_feat = state_features(port_feat, rt)
    print(f"  states:   {len(state_feat):,} rows  (state × month-year)")

    # Merge concentration onto port_feat for a single port-feature file.
    port_feat.to_parquet(PROCESSED / "port_network_features.parquet", index=False, compression="zstd")
    state_feat.to_parquet(PROCESSED / "state_network_features.parquet", index=False, compression="zstd")
    cc_feat.to_parquet(PROCESSED / "country_concentration.parquet", index=False, compression="zstd")

    # Quick descriptive snapshot
    print("\nTop 10 US ports by Shannon entropy of inbound origin countries (2025-10):")
    snap = port_feat[(port_feat["year"] == 2025) & (port_feat["month"] == 10)]
    print(snap.nlargest(10, "port_origin_entropy")[
        ["dest_us_port", "port_n_countries", "port_origin_entropy", "port_total_pax",
         "port_inf_share_capitata", "port_inf_share_dorsalis", "port_inf_share_ludens"]
    ].to_string(index=False, float_format="%.3f"))

    print("\nTop 10 states by infested-passenger SHARE for Bactrocera dorsalis (2025-10):")
    sd = state_feat[(state_feat["year"] == 2025) & (state_feat["month"] == 10)]
    print(sd.nlargest(10, "state_inf_share_dorsalis")[
        ["state", "state_n_countries", "state_origin_entropy", "state_inf_share_dorsalis", "state_total_pax"]
    ].to_string(index=False, float_format="%.3f"))

    print("\nWrote data/processed/port_network_features.parquet")
    print("Wrote data/processed/state_network_features.parquet")
    print("Wrote data/processed/country_concentration.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
