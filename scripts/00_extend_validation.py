"""
Extend the APHIS validation set with additional detection events.

The current validation set has 77 events from APHIS PDR (federal-order detections).
This is the binding constraint on backtest precision and on the model's calibration
on rare cells. Adding 100-500 events from CDFA + FDACS state-level detection
bulletins would dramatically tighten the model.

This script accepts an external CSV in our schema and merges it into the
validation file. Use it when you receive data from:

  - CDFA Plant Pest Diagnostics — Mediterranean / Mexican / Oriental fruit fly
    detection records by year, county, species. Available via FOIA or partnership;
    historical records back to ~2000.
  - FDACS Plant Industry — Florida fruit fly detection bulletins. The Caribbean
    fruit fly (A. suspensa) is established in FL, so detection records there are
    routine — would massively expand the suspensa label set.
  - APHIS AQAS port-of-entry interception database — internal to USDA APHIS;
    requires FOIA or program-partner access.
  - 2014 Szyniszewska & Tatem PLOS One paper — supplementary tables list
    historical port-level Mediterranean fruit fly interceptions; useful for
    pre-2018 backfill.
  - State-level annual reports (CDFA Annual Report, FDACS Tri-Annual Plant Pest
    Report) — published PDFs with detection summaries.

Required CSV schema (matches data/raw/pests/aphis_validation.csv):
  year, month, state_or_port, species, count, source, source_url, notes

  state_or_port: 2-letter state code (CA, FL, TX, NY, ...) or full IATA airport code
  species: full scientific name (e.g. "Ceratitis capitata")
  count: integer number of detection events for that (state, month, species, source)
  source: short identifier of the data origin (e.g. "CDFA_2024_annual")
  source_url: where you got it
  notes: any additional metadata

Usage:
  python scripts/00_extend_validation.py path/to/external_events.csv
  # validates schema, dedupes against existing, appends, prints summary

After extension, re-run the pipeline to refit the model with the larger labels:
  python scripts/02_build_join_table.py
  python scripts/05_network_features.py
  python scripts/03_fit_risk_model.py
  python scripts/07_county_predict.py
  python scripts/08_backtest.py        # backtest precision improves with N
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
VALIDATION_PATH = RAW / "pests" / "aphis_validation.csv"

REQUIRED_COLS = ["year", "month", "state_or_port", "species", "count",
                 "source", "source_url", "notes"]

# Species names we currently model. Other species in the input get warned but kept;
# they won't drive the model fit until the species is added to the pipeline.
MODELED_SPECIES = {
    "Ceratitis capitata", "Bactrocera dorsalis", "Anastrepha ludens",
    "Anastrepha suspensa", "Bactrocera zonata",   "Rhagoletis cerasi",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}\nRequired schema: {REQUIRED_COLS}")
    bad_year = df[~df["year"].between(1990, 2030)]
    if not bad_year.empty:
        sys.exit(f"{len(bad_year)} rows have year outside [1990, 2030] — typo?")
    bad_month = df[~df["month"].between(1, 12)]
    if not bad_month.empty:
        sys.exit(f"{len(bad_month)} rows have month outside [1, 12]")
    bad_state = df[df["state_or_port"].astype(str).str.len() > 4]
    if not bad_state.empty:
        sys.exit(f"{len(bad_state)} rows have state_or_port longer than 4 chars — "
                 "use 2-letter state codes (CA) or 3-letter IATA codes (LAX), "
                 "not full names")


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <external_events.csv>")
    ext_path = Path(sys.argv[1])
    if not ext_path.exists():
        sys.exit(f"File not found: {ext_path}")

    new = pd.read_csv(ext_path)
    print(f"Loaded {len(new):,} rows from {ext_path}")
    validate_schema(new)

    existing = pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else pd.DataFrame(columns=REQUIRED_COLS)
    print(f"Existing validation rows: {len(existing):,}")

    # Dedupe by (year, month, state_or_port, species, source) — same source
    # reporting same event is one fact, not two.
    combined = pd.concat([existing, new], ignore_index=True)
    dedup_keys = ["year", "month", "state_or_port", "species", "source"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
    after = len(combined)
    print(f"Dropped {before - after:,} duplicate rows  →  {after:,} total")

    # Warn on unmodeled species
    sp_in_new = set(new["species"].unique())
    new_only = sp_in_new - MODELED_SPECIES
    if new_only:
        print(f"\nWarning — {len(new_only)} species not yet modeled by PESTCAST:")
        for sp in sorted(new_only):
            print(f"  {sp}")
        print("  These rows are kept in the validation file but won't influence the "
              "model until the species is added to scripts/02, 03, 05, 06, 07.")

    # Backup before writing
    backup = VALIDATION_PATH.with_suffix(".csv.bak")
    if VALIDATION_PATH.exists():
        backup.write_bytes(VALIDATION_PATH.read_bytes())
        print(f"Backed up existing file to {backup.name}")

    combined.to_csv(VALIDATION_PATH, index=False)
    print(f"\nWrote {VALIDATION_PATH} with {len(combined):,} rows")

    print("\nSummary by year:")
    print(combined.groupby("year").size().to_string())
    print("\nSummary by species (top 10):")
    print(combined["species"].value_counts().head(10).to_string())

    print("\nNext step: re-run the pipeline so the model fits on the expanded labels:")
    print("  python scripts/02_build_join_table.py")
    print("  python scripts/05_network_features.py")
    print("  python scripts/03_fit_risk_model.py")
    print("  python scripts/07_county_predict.py")
    print("  python scripts/08_backtest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
