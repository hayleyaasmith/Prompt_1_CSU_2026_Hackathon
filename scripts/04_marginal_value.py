"""
Fruit Fly Risk — Marginal value of inspection.

Converts the fitted Poisson predictions into the operational deliverable:
"where does the next inspector-hour catch the most pests, and what would it cost
to get it from somewhere else?"

Detection model (the layer the regression doesn't capture):
    Each cell (state × month) has a latent arrival rate of infested material A_i,
    and h_i inspector-hours allocated. Detection follows an exponential capture:
        p_detect_i  = 1 - exp(-h_i / k)
        observed μ  = A_i * p_detect_i      (matches the fitted Poisson μ)
    so the latent arrival rate is recovered as A_i = μ_i / p_detect_i.
    Marginal value of one additional hour:
        dμ/dh = (A_i / k) * (1 - p_detect_i)
    which formalizes diminishing returns: cells already heavily inspected gain less.

Two free parameters are exposed at the top of the file. Both are tunable;
neither changes the cell *ranking*, only the absolute scale of the recommendations:
    TOTAL_HOURS         total annual US-PPQ inspector-hours, all cells combined.
    BASELINE_DETECT_PROB the per-arrival detection probability at a cell with
                         average hours. Sets k.

Baseline policy assumed in this script:
    Hours allocated *pro-rata to passenger volume from infested countries*.
    This is a reasonable proxy for current CBP/PPQ practice and the right
    counterfactual to compare a smarter allocation against. Replace this with
    real PPQ staffing data when available.

Outputs:
    data/processed/marginal_value.parquet            per (state, month) cell
    data/processed/reallocation_recommendations.csv  top transfer suggestions
    data/processed/cell_species_breakdown.parquet    species-level diagnostic

Usage:
    python scripts/04_marginal_value.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# --- Tunable constants ------------------------------------------------------
TOTAL_HOURS          = 600_000   # notional total US-PPQ port-of-entry inspector-hours/year
BASELINE_DETECT_PROB = 0.05      # per-arrival detection probability at an average cell
SHIFT_HOURS          = 200       # block size for recommendation rows
N_RECOMMENDATIONS    = 15
# ---------------------------------------------------------------------------


def aggregate_cells(pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregate species-level predictions to (state, month). Inspections are
    species-agnostic; species split is preserved separately for diagnostics."""
    return (pred.groupby(["state", "month"], as_index=False)
                .agg(mu=("mu", "sum"),
                     mu_lo80=("mu_lo80", "sum"),
                     mu_hi80=("mu_hi80", "sum"),
                     detections=("detections", "sum"),
                     inf_pass=("inf_pass", "sum"),
                     inf_freight=("inf_freight", "sum"),
                     inf_host=("inf_host", "sum")))


def compute_marginal_value(cells: pd.DataFrame) -> pd.DataFrame:
    """Apply the exponential capture model and compute marginal value per hour."""
    cells = cells.copy()
    total_pass = cells["inf_pass"].sum()
    if total_pass <= 0:
        sys.exit("All cells have zero infested-passenger volume — cannot allocate.")
    cells["pass_share"] = cells["inf_pass"] / total_pass
    cells["hours"]      = TOTAL_HOURS * cells["pass_share"]

    # k chosen so that a cell with average passenger share gets p_detect = BASELINE_DETECT_PROB.
    avg_hours = TOTAL_HOURS / len(cells)
    k = -avg_hours / np.log(1 - BASELINE_DETECT_PROB)

    cells["p_detect"]        = 1 - np.exp(-cells["hours"] / k)
    cells["latent_arrivals"] = np.where(cells["p_detect"] > 0,
                                        cells["mu"] / cells["p_detect"], 0.0)
    cells["marginal_per_hour"] = (cells["latent_arrivals"] / k) * (1 - cells["p_detect"])

    # Useful diagnostics
    cells["k"]                = k
    cells["avg_hours_per_cell"] = avg_hours
    return cells


def reallocation_recommendations(cells: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Pair the highest-marginal-value cells with the lowest, with
    a per-block expected-gain estimate (linear approximation, valid for small shifts)."""
    sp = pred.groupby(["state", "month", "species"], as_index=False)["mu"].sum()
    top_species = (sp.sort_values("mu", ascending=False)
                     .drop_duplicates(subset=["state", "month"])
                     .set_index(["state", "month"])["species"])

    receivers = (cells.sort_values("marginal_per_hour", ascending=False)
                       .reset_index(drop=True))
    # Donors must have enough hours to actually shift; rank lowest-MV among those.
    donors = (cells[cells["hours"] >= SHIFT_HOURS]
                .sort_values("marginal_per_hour", ascending=True)
                .reset_index(drop=True))

    n = min(N_RECOMMENDATIONS, len(receivers), len(donors))
    recs = []
    for i in range(n):
        to_   = receivers.iloc[i]
        from_ = donors.iloc[i]
        if (to_["state"], to_["month"]) == (from_["state"], from_["month"]):
            continue  # don't shift to/from the same cell
        gain = SHIFT_HOURS * (to_["marginal_per_hour"] - from_["marginal_per_hour"])
        if gain <= 0:
            break
        recs.append({
            "rank":               i + 1,
            "shift_hours":        SHIFT_HOURS,
            "from_state":         from_["state"],
            "from_month":         int(from_["month"]),
            "from_marginal_per_h": round(from_["marginal_per_hour"], 6),
            "to_state":           to_["state"],
            "to_month":           int(to_["month"]),
            "to_marginal_per_h":  round(to_["marginal_per_hour"], 6),
            "to_dominant_species": top_species.get((to_["state"], to_["month"]), ""),
            "expected_gain_detections": round(gain, 4),
        })
    return pd.DataFrame(recs)


def report(cells: pd.DataFrame, recs: pd.DataFrame) -> None:
    print("\n=== Marginal value of inspection — by cell ===")
    print(f"Cells:                 {len(cells)}")
    print(f"Total μ (predicted):   {cells['mu'].sum():.2f}  detections/yr at current allocation")
    print(f"Total latent arrivals: {cells['latent_arrivals'].sum():.0f}  (μ / p_detect)")
    print(f"k (saturation const):  {cells['k'].iloc[0]:.1f} hours")
    print(f"Average hours/cell:    {cells['avg_hours_per_cell'].iloc[0]:.0f}")

    print("\nTop 10 cells by MARGINAL VALUE (where the next hour catches the most):")
    cols = ["state", "month", "mu", "hours", "p_detect", "marginal_per_hour"]
    print(cells.sort_values("marginal_per_hour", ascending=False)
               .head(10)[cols]
               .to_string(index=False, float_format="%.4f"))

    print("\nBottom 10 cells by MARGINAL VALUE (cheap places to redeploy from):")
    print(cells.sort_values("marginal_per_hour", ascending=True)
               .head(10)[cols]
               .to_string(index=False, float_format="%.4f"))

    print(f"\nReallocation recommendations (each row: shift {SHIFT_HOURS} hours):")
    if recs.empty:
        print("  (no positive-gain transfers found at this shift size)")
    else:
        cols_rec = ["rank", "from_state", "from_month", "to_state", "to_month",
                    "to_dominant_species", "expected_gain_detections"]
        print(recs[cols_rec].to_string(index=False, float_format="%.4f"))
        total_gain = recs["expected_gain_detections"].sum()
        baseline   = cells["mu"].sum()
        print(f"\nIf all {len(recs)} transfers were executed:")
        print(f"  Total hours redeployed: {recs['shift_hours'].sum():,}  ({recs['shift_hours'].sum()/TOTAL_HOURS*100:.2f}% of budget)")
        print(f"  Total expected gain:    {total_gain:.2f} detections")
        print(f"  Relative to baseline:   +{total_gain/baseline*100:.1f}%")


def main() -> int:
    pred = pd.read_parquet(PROCESSED / "risk_predictions.parquet")
    cells = aggregate_cells(pred)
    cells = compute_marginal_value(cells)
    recs  = reallocation_recommendations(cells, pred)
    report(cells, recs)

    cells.to_parquet(PROCESSED / "marginal_value.parquet", index=False, compression="zstd")
    recs.to_csv(PROCESSED / "reallocation_recommendations.csv", index=False)

    breakdown = (pred.merge(cells[["state", "month", "marginal_per_hour", "p_detect"]],
                            on=["state", "month"], how="left"))
    breakdown.to_parquet(PROCESSED / "cell_species_breakdown.parquet", index=False, compression="zstd")

    print(f"\nWrote data/processed/marginal_value.parquet              ({len(cells)} rows)")
    print(f"Wrote data/processed/reallocation_recommendations.csv    ({len(recs)} rows)")
    print(f"Wrote data/processed/cell_species_breakdown.parquet      ({len(breakdown)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
