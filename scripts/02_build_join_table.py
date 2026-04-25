"""
Fruit Fly Risk — Build the canonical join table.

Grain: one row per (origin_country, dest_us_port, year, month).
Columns:
    passengers, freight_kg, mail_kg                    — from BTS T-100
    host_kg_total                                      — from USDA FAS GATS
    present_capitata, present_dorsalis, present_ludens — from EPPO
    any_pest_present                                   — derived
    infested_passengers, infested_freight_kg           — pathway × pest interactions

Notes / known limitations of v1:
    - GATS reports imports by partner country only, not by U.S. port. We carry
      it through unallocated; v2 will allocate via airport / customs district.
    - EPPO presence is country-level and time-invariant. Seasonal phenology
      (peak months per species) will be layered in by 03_score_risk.py.
    - Connecting flights are NOT yet attributed to true origin (T-100 only
      sees the last segment). 04_network_flow.py handles that.

Usage:
    python scripts/02_build_join_table.py
Output:
    data/processed/risk_table.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Fruit fly host commodities (HS 4-digit prefixes from DATA_PLAN.md).
HOST_HS4 = {"0805", "0804", "0807", "0809", "0810", "0702", "0707"}

# EPPO "Present" statuses we treat as established presence. Excludes
# Transient (interceptions only) and all "Absent, *" categories.
PRESENT_STATUSES = {
    "Present, no details",
    "Present, restricted distribution",
    "Present, widespread",
    "Present, few occurrences",
}

EPPO_FILES = {
    "capitata":  RAW / "pests" / "eppo_ceratitis_capitata.csv",
    "dorsalis":  RAW / "pests" / "eppo_bactrocera_dorsalis.csv",
    "ludens":    RAW / "pests" / "eppo_anastrepha_ludens.csv",
    "suspensa":  RAW / "pests" / "eppo_anastrepha_suspensa.csv",
    "zonata":    RAW / "pests" / "eppo_bactrocera_zonata.csv",
    "cerasi":    RAW / "pests" / "eppo_rhagoletis_cerasi.csv",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_t100() -> pd.DataFrame:
    """T-100 international segments → (origin_country, dest_us_port, year, month, passengers, freight_kg, mail_kg)."""
    files = sorted((RAW / "trade").glob("t100_international_*.csv"))
    if not files:
        sys.exit("No T-100 files found in data/raw/trade/")
    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            usecols=["PASSENGERS", "FREIGHT", "MAIL", "ORIGIN_COUNTRY",
                     "DEST", "DEST_COUNTRY", "YEAR", "MONTH"],
            dtype={"ORIGIN_COUNTRY": "string", "DEST": "string", "DEST_COUNTRY": "string"},
        )
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # Inbound to US, truly international.
    mask = (raw["DEST_COUNTRY"] == "US") & (raw["ORIGIN_COUNTRY"] != "US") & raw["ORIGIN_COUNTRY"].notna()
    raw = raw.loc[mask].copy()

    # T-100 reports FREIGHT/MAIL in pounds; convert to kg for parity with GATS.
    LB_TO_KG = 0.45359237
    agg = (
        raw.groupby(["ORIGIN_COUNTRY", "DEST", "YEAR", "MONTH"], as_index=False, observed=True)
           .agg(passengers=("PASSENGERS", "sum"),
                freight_lb=("FREIGHT", "sum"),
                mail_lb=("MAIL", "sum"))
    )
    agg["freight_kg"] = agg["freight_lb"] * LB_TO_KG
    agg["mail_kg"]    = agg["mail_lb"]    * LB_TO_KG
    agg = agg.drop(columns=["freight_lb", "mail_lb"])
    agg = agg.rename(columns={
        "ORIGIN_COUNTRY": "origin_country",
        "DEST": "dest_us_port",
        "YEAR": "year",
        "MONTH": "month",
    })
    return agg


def load_gats() -> pd.DataFrame:
    """GATS host imports → (origin_country, year, month, host_kg_total)."""
    g = pd.read_csv(RAW / "trade" / "gats_host_imports.csv", low_memory=False)
    g["hs4"] = g["HS Code"].astype("Int64").astype(str).str.zfill(10).str[:4]
    g = g[g["hs4"].isin(HOST_HS4)].copy()

    # Year column is "2025-2025"; take the first half.
    g["year"] = g["Year"].str.split("-").str[0].astype(int)

    # Only keep KG units (drop NUMBER/L/etc. — small).
    g = g[g["UOM"] == "KG"].copy()

    qty_cols = [f"{m}_Qty" for m in MONTHS]
    keep = ["Total_Partner Code", "year"] + qty_cols
    g = g[keep].copy()

    # Quantity cells are strings with thousands separators. Normalize.
    for c in qty_cols:
        g[c] = pd.to_numeric(g[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    long = g.melt(
        id_vars=["Total_Partner Code", "year"],
        value_vars=qty_cols,
        var_name="month_name",
        value_name="host_kg",
    )
    long["month"] = long["month_name"].str.replace("_Qty", "", regex=False).map({m: i + 1 for i, m in enumerate(MONTHS)})
    long = long.drop(columns=["month_name"])
    long = long.rename(columns={"Total_Partner Code": "origin_country"})

    out = (
        long.dropna(subset=["host_kg", "origin_country"])
            .groupby(["origin_country", "year", "month"], as_index=False)
            .agg(host_kg_total=("host_kg", "sum"))
    )
    return out


def load_eppo() -> pd.DataFrame:
    """EPPO presence → wide table indexed by ISO-2 country code."""
    rows = []
    for species, path in EPPO_FILES.items():
        df = pd.read_csv(path)
        df = df[df["state code"].isna() | (df["state code"] == "")].copy()  # drop subnational rows
        df["present"] = df["Status"].isin(PRESENT_STATUSES).astype(int)
        df = df[["country code", "present"]].rename(columns={"country code": "origin_country"})
        df["species"] = species
        rows.append(df)
    long = pd.concat(rows, ignore_index=True)
    wide = (
        long.pivot_table(index="origin_country", columns="species", values="present",
                         aggfunc="max", fill_value=0)
            .reset_index()
    )
    # Add present_<species> prefix without affecting origin_country
    wide.columns = ["origin_country"] + [f"present_{c}" for c in wide.columns if c != "origin_country"]
    wide["any_pest_present"] = wide.filter(like="present_").max(axis=1).astype(int)
    return wide


def build() -> pd.DataFrame:
    print("Loading T-100…")
    t100 = load_t100()
    print(f"  {len(t100):,} (origin_country × dest_us_port × year × month) rows")

    print("Loading GATS host imports…")
    gats = load_gats()
    print(f"  {len(gats):,} (origin_country × year × month) rows  |  {gats['origin_country'].nunique()} partners")

    print("Loading EPPO pest presence…")
    eppo = load_eppo()
    n_inf = int(eppo["any_pest_present"].sum())
    print(f"  {len(eppo):,} countries  |  {n_inf} with at least one target species present")

    print("Joining…")
    out = t100.merge(gats, on=["origin_country", "year", "month"], how="left")
    out = out.merge(eppo, on="origin_country", how="left")

    # Countries unknown to EPPO → assume absent (conservative: treats as no risk).
    presence_cols = ["present_capitata", "present_dorsalis", "present_ludens", "any_pest_present"]
    out[presence_cols] = out[presence_cols].fillna(0).astype(int)

    # Pathway × pest interactions — the actual risk-relevant volumes.
    out["host_kg_total"] = out["host_kg_total"].fillna(0.0)
    out["infested_passengers"] = np.where(out["any_pest_present"] == 1, out["passengers"], 0.0)
    out["infested_freight_kg"] = np.where(out["any_pest_present"] == 1, out["freight_kg"], 0.0)

    # Drop network features columns that depend on a fixed set of species.
    # Downstream scripts (network features, model fit) recompute as needed.

    return out


def report(df: pd.DataFrame) -> None:
    # host_kg_total is country-month granularity (GATS doesn't break out by port),
    # so it's replicated across every port row for the same origin-month. Dedupe
    # before summing to avoid the (#ports * actual) inflation.
    host_dedup = (df[["origin_country", "year", "month", "host_kg_total"]]
                    .drop_duplicates(["origin_country", "year", "month"]))

    print("\n=== Join table summary ===")
    print(f"Rows:                 {len(df):,}")
    print(f"Origin countries:     {df['origin_country'].nunique():,}")
    print(f"US ports of entry:    {df['dest_us_port'].nunique():,}")
    print(f"Years:                {sorted(df['year'].unique())}")
    print(f"Total passengers:     {df['passengers'].sum():,.0f}")
    print(f"Total freight (kg):   {df['freight_kg'].sum():,.0f}")
    print(f"Total host imp (kg):  {host_dedup['host_kg_total'].sum():,.0f}  (country-month, deduped)")
    coverage = (df['any_pest_present'] == 1).mean() * 100
    print(f"Rows with pest in origin country:  {coverage:.1f}%")

    print("\nTop 10 (origin × port × month) by infested passenger volume:")
    cols = ["origin_country", "dest_us_port", "year", "month",
            "passengers", "any_pest_present", "infested_passengers", "host_kg_total"]
    top = (df.sort_values("infested_passengers", ascending=False)
             .head(10)[cols]
             .to_string(index=False))
    print(top)


def main() -> int:
    df = build()
    report(df)
    out_path = PROCESSED / "risk_table.parquet"
    df.to_parquet(out_path, index=False, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
