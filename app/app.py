"""
PESTCAST — Pathway Pest Risk Intelligence
USDA APHIS PPQ surveillance dashboard.

Run:
    .venv/bin/streamlit run app/app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SPECIES_LABEL = {
    "capitata": "Mediterranean fruit fly (C. capitata)",
    "dorsalis": "Oriental fruit fly (B. dorsalis)",
    "ludens":   "Mexican fruit fly (A. ludens)",
    "suspensa": "Caribbean fruit fly (A. suspensa)",
    "zonata":   "Peach fruit fly (B. zonata)",
    "cerasi":   "European cherry fruit fly (R. cerasi)",
}
SPECIES_LIST = list(SPECIES_LABEL.keys())
SPECIES_SHORT = {k: v.split(" (")[0] for k, v in SPECIES_LABEL.items()}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Risk-tier thresholds (fraction of top combined risk in the panel) used for
# operational status badges in tables.
TIER_HIGH   = 0.50
TIER_MED    = 0.20

VERSION = "0.5.0"
FORECAST_YEAR = 2026
BASELINE_YEAR = 2025  # year of features used to project the forecast year
DATA_VINTAGE = (f"Forecast: CY{FORECAST_YEAR} · features: CY{BASELINE_YEAR} · "
                "T-100 2015-2025 · GATS 2015-2026 · EPPO 2024-12-10 · APHIS PDR through 2026-04")

# --- Forecast horizon: only the next ~90 days are operationally reliable ----
TODAY = datetime.now().date()
HORIZON_DAYS = 90
HORIZON_END = TODAY + timedelta(days=HORIZON_DAYS)

def _near_term_months(today: "datetime.date") -> set[int]:
    """Months falling within the operational horizon (current month + ~90 days
    forward, capped at calendar year-end since the forecast year is fixed)."""
    months = set()
    cur = today.replace(day=1)
    end_marker = today + timedelta(days=HORIZON_DAYS)
    while cur <= end_marker:
        if cur.year == FORECAST_YEAR:
            months.add(cur.month)
        # advance to the first of next month
        nxt_year = cur.year + (1 if cur.month == 12 else 0)
        nxt_month = 1 if cur.month == 12 else cur.month + 1
        cur = cur.replace(year=nxt_year, month=nxt_month)
    return months or {today.month}

NEAR_TERM_MONTHS = _near_term_months(TODAY)
NEAR_TERM_LABEL = (f"{MONTH_NAMES[min(NEAR_TERM_MONTHS) - 1]}–"
                   f"{MONTH_NAMES[max(NEAR_TERM_MONTHS) - 1]} {FORECAST_YEAR}")
HORIZON_LABEL_SHORT = f"{TODAY.strftime('%b %d')} → {HORIZON_END.strftime('%b %d')}"


# ---------------------------------------------------------------------------
# Page config + custom styling (light-mode locked)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PESTCAST — Pathway Pest Risk Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": None,
        "About": "PESTCAST — Pathway Pest Risk Intelligence. Geospatial pathway risk model "
                 "for USDA APHIS PPQ. Built on T-100, GATS, EPPO, WorldClim, and APHIS PDR.",
    },
)

st.markdown("""
<style>
/* hide streamlit chrome */
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden; height: 0;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

/* typography */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
    color: #0f172a;
}
h1 { font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.02em; margin: 0 !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.01em; margin: 0.6rem 0 0.4rem 0 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; letter-spacing: -0.01em; color: #334155; margin: 0.5rem 0 0.3rem 0 !important; }

/* page background */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
}
[data-testid="stMain"] .block-container {
    background-color: #f8fafc;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* metric cards (st.metric) */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    min-height: 88px;
}
[data-testid="stMetricLabel"] {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.2;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a;
    font-size: 1.25rem !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
[data-testid="stMetricDelta"] {
    color: #475569 !important;
    font-size: 0.74rem;
    line-height: 1.3;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* card class for content blocks */
.surveillance-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    margin-bottom: 12px;
}

/* dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

/* tabs */
[data-baseweb="tab-list"] {
    gap: 2px;
    background: transparent;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0;
    margin-bottom: 1rem;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 14px !important;
    font-weight: 500;
    font-size: 0.9rem;
    height: auto !important;
    border-radius: 0 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #15803d !important;
    border-bottom: 2px solid #15803d !important;
    background: transparent !important;
}

/* status pills */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.pill-critical { background: #fee2e2; color: #991b1b; }
.pill-elevated { background: #fed7aa; color: #9a3412; }
.pill-watch    { background: #fef3c7; color: #92400e; }
.pill-routine  { background: #dcfce7; color: #166534; }

/* header bar */
.app-header {
    background: linear-gradient(90deg, #14532d 0%, #166534 100%);
    color: #ecfccb;
    padding: 14px 24px;
    border-radius: 0;
    margin: -1.5rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #84cc16;
}
.app-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.app-brand-mark {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(132, 204, 22, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: "Iowan Old Style", "Palatino", Georgia, serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: #ecfccb;
    letter-spacing: 0;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
}
.app-brand-text {
    display: flex;
    flex-direction: column;
    line-height: 1.15;
}
.app-brand-name {
    font-family: "Iowan Old Style", "Palatino", Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    color: #ffffff;
    margin: 0;
}
.app-brand-tagline {
    font-size: 0.73rem;
    color: #bbf7d0;
    letter-spacing: 0.04em;
    font-weight: 500;
    text-transform: uppercase;
}
.app-brand-org {
    font-size: 0.7rem;
    color: rgba(187, 247, 208, 0.7);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}
.app-header-meta {
    font-size: 0.78rem;
    color: #bbf7d0;
    font-variant-numeric: tabular-nums;
    text-align: right;
    line-height: 1.5;
}
.app-header-meta-version {
    font-family: "SF Mono", "Menlo", Consolas, monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.04em;
}
.app-header-horizon {
    background: rgba(132, 204, 22, 0.18);
    border: 1px solid rgba(132, 204, 22, 0.55);
    color: #ecfccb;
    padding: 4px 10px 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 4px;
}

/* near-term ribbon CSS removed — replaced with inline tier badges */

/* outlook treatment (months past horizon) */
.outlook-section {
    opacity: 0.85;
    border-top: 1px dashed #cbd5e1;
    margin-top: 18px;
    padding-top: 12px;
}
.outlook-eyebrow {
    display: inline-block;
    background: #f1f5f9;
    color: #64748b;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

/* divider */
hr { margin: 1rem 0; border-color: #e2e8f0 !important; }

/* st.pills polish — used for unified species selector */
[data-testid="stPills"] button {
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
    border-radius: 999px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #475569 !important;
    font-weight: 500 !important;
    margin: 2px !important;
    transition: all 0.12s ease;
}
[data-testid="stPills"] button:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}
[data-testid="stPills"] button[aria-pressed="true"] {
    background: #15803d !important;
    color: #ffffff !important;
    border-color: #15803d !important;
}
[data-testid="stPills"] button[aria-pressed="true"]:hover {
    background: #166534 !important;
    border-color: #166534 !important;
}

/* tighter buttons (used for select-all/none) */
[data-testid="stBaseButton-secondary"] {
    background: #ffffff;
    color: #475569;
    border: 1px solid #cbd5e1;
    font-size: 0.74rem;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 500;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
    color: #0f172a;
}

/* selectbox / radio polish */
[data-testid="stRadio"] label { font-weight: 500; }

/* expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #ffffff;
}

/* caption */
[data-testid="stCaptionContainer"] {
    color: #64748b;
    font-size: 0.82rem;
}

/* tighten row gaps */
[data-testid="column"] { padding: 0 0.6rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

/* section headings inside tabs */
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 18px 0 10px 0;
}
.section-meta {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    margin-left: 8px;
}

/* informational pills (replaces big yellow callouts) */
.info-strip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px 4px 8px;
    background: #fef9c3;
    border: 1px solid #fde68a;
    border-radius: 6px;
    font-size: 0.77rem;
    color: #713f12;
    margin-bottom: 10px;
}
.info-strip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ca8a04;
}

/* download button polish */
[data-testid="stDownloadButton"] button {
    background: #ffffff;
    color: #475569;
    border: 1px solid #cbd5e1;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 6px;
}
[data-testid="stDownloadButton"] button:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "risk_predictions.parquet")


@st.cache_data
def load_marginal() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "marginal_value.parquet")


@st.cache_data
def load_coefficients() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "model_coefficients.csv")


@st.cache_data
def load_breakdown() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "cell_species_breakdown.parquet")


@st.cache_data
def load_state_network() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "state_network_features.parquet")


@st.cache_data
def load_risk_table() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "risk_table.parquet")


@st.cache_data
def load_climate_county() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "climate_suitability_by_county.parquet")


@st.cache_data
def load_counties_geojson() -> dict:
    with open(RAW / "geo" / "us_counties.geojson") as f:
        return json.load(f)


@st.cache_data
def load_county_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "county_predictions.parquet")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_county_history() -> pd.DataFrame:
    """5-year history of county-level annual predictions for sparklines."""
    df = pd.read_parquet(PROCESSED / "county_predictions_history.parquet")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df


@st.cache_data
def yoy_sparkline_dict(species: str) -> dict:
    """Return {fips: [year-by-year combined risk]} for sparkline embedding."""
    h = load_county_history()
    sub = h[h["species"] == species].sort_values(["fips", "year"])
    return {f: sub.loc[sub["fips"] == f, "annual_combined"].tolist()
            for f in sub["fips"].unique()}


@st.cache_data
def county_country_drivers(species: str, fips: str, year: int = BASELINE_YEAR) -> pd.DataFrame:
    """For a given county, the top origin countries driving its pathway risk.
    Computed at the (county, origin) granularity by joining risk_table with
    the airport→county map and filtering to airports inside the county."""
    rt = load_risk_table()
    amap = pd.read_parquet(PROCESSED / "airport_county_map.parquet")
    amap = amap.rename(columns={"iata_code": "dest_us_port"})
    amap["fips"] = amap["fips"].astype(str).str.zfill(5)

    rt_year = rt[rt["year"] == year].copy()
    rt_year = rt_year[rt_year[f"present_{species}"] == 1]
    rt_with_co = rt_year.merge(amap[["dest_us_port", "fips"]], on="dest_us_port", how="inner")
    rt_with_co = rt_with_co[rt_with_co["fips"] == fips]

    drivers = (rt_with_co.groupby("origin_country", as_index=False)
                          .agg(passengers=("passengers", "sum"),
                               freight_kg=("freight_kg", "sum")))
    if drivers.empty:
        return drivers
    drivers["share"] = drivers["passengers"] / drivers["passengers"].sum()
    return drivers.sort_values("passengers", ascending=False)


@st.cache_data
def multi_species_hotspots(top_n: int = 20) -> pd.DataFrame:
    """Counties that appear in top-N annual rankings for ≥ 2 species."""
    cp = load_county_predictions()
    annual = (cp.groupby(["fips", "state", "county_name", "species"], as_index=False)
                .agg(annual_combined=("combined", "sum")))
    rows = []
    for sp, grp in annual.groupby("species"):
        top = grp.nlargest(top_n, "annual_combined")
        for _, r in top.iterrows():
            rows.append({"fips": r["fips"], "state": r["state"],
                         "county_name": r["county_name"],
                         "species": sp, "annual_combined": r["annual_combined"]})
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    counts = (long.groupby(["fips", "state", "county_name"])
                  .agg(n_species=("species", "nunique"),
                       species_list=("species", lambda s: ", ".join(sorted(set(s)))),
                       total_combined=("annual_combined", "sum"))
                  .reset_index())
    return counts[counts["n_species"] >= 2].sort_values(
        ["n_species", "total_combined"], ascending=[False, False])


@st.cache_data
def load_county_centroids() -> pd.DataFrame:
    """Load pre-computed county centroids (lon, lat) from parquet file, keyed by FIPS.
    Centroids were computed using an equal-area projection for numerical stability,
    then projected back to WGS84 for plotting."""
    return pd.read_parquet(PROCESSED / "county_centroids.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tier(value: float, top_value: float) -> str:
    """Classify a risk value into a tier badge based on its share of the panel max."""
    if top_value <= 0:
        return "routine"
    frac = value / top_value
    if frac >= TIER_HIGH:
        return "critical"
    if frac >= TIER_MED:
        return "elevated"
    if frac >= 0.05:
        return "watch"
    return "routine"


def tier_label(t: str) -> str:
    return {
        "critical": "Critical",
        "elevated": "Elevated",
        "watch":    "Watch",
        "routine":  "Routine",
    }[t]


def csv_button(df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def professional_choropleth(df: pd.DataFrame, geojson: dict, color_col: str,
                            color_label: str, range_color, hover_data,
                            colorscale: str = "Reds") -> go.Figure:
    """Standardized county choropleth with light-mode geo styling."""
    fig = px.choropleth(
        df, geojson=geojson, locations="fips",
        color=color_col,
        color_continuous_scale=colorscale,
        range_color=range_color,
        scope="usa",
        hover_data=hover_data,
        labels={color_col: color_label},
    )
    fig.update_traces(marker_line_width=0.2, marker_line_color="rgba(71, 85, 105, 0.25)")
    fig.update_geos(
        showsubunits=True,
        subunitcolor="rgba(71, 85, 105, 0.5)",
        subunitwidth=0.6,
        showland=True, landcolor="#f1f5f9",
        showcoastlines=True, coastlinecolor="rgba(71, 85, 105, 0.4)",
        coastlinewidth=0.6,
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
        lakecolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(thickness=12, len=0.7, outlinewidth=0,
                                tickfont=dict(size=10, color="#475569")),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif",
                  size=12, color="#0f172a"),
    )
    return fig


def style_chart(fig: go.Figure) -> go.Figure:
    """Apply consistent professional styling to non-geo plotly figures."""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif",
                  size=12, color="#0f172a"),
        margin=dict(l=8, r=8, t=24, b=8),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", tickfont=dict(size=11, color="#475569")),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", tickfont=dict(size=11, color="#475569")),
    )
    return fig


def _build_briefing_html(species: str, month: int, annual_sp: pd.DataFrame,
                         snap_active: pd.DataFrame, hotspots: pd.DataFrame) -> str:
    """Generate a printable HTML briefing for leadership distribution."""
    sp_label = SPECIES_LABEL.get(species, species)
    sp_short = SPECIES_SHORT.get(species, species)
    month_name = MONTH_NAMES[month - 1]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    top_annual = annual_sp.nlargest(10, "annual_combined").copy()
    top_monthly = snap_active.nlargest(10, "combined").copy() if not snap_active.empty else pd.DataFrame()

    def _row(name, state, *vals):
        cells = "".join(f"<td>{v}</td>" for v in vals)
        return f"<tr><td>{name}</td><td>{state}</td>{cells}</tr>"

    annual_rows = "".join(
        _row(r["county_name"], r["state"],
             f"{r['annual_combined']:.2f}",
             f"{r['annual_pathway']:.2f}",
             f"{r['climate_frac']:.2f}")
        for _, r in top_annual.iterrows()
    ) if not top_annual.empty else "<tr><td colspan='5'>No active counties</td></tr>"

    has_ci = {"combined_lo80", "combined_hi80"}.issubset(top_monthly.columns)
    monthly_rows = "".join(
        _row(r["county_name"], r["state"],
             f"{r['combined']:.3f}",
             (f"{r['combined_lo80']:.3f}–{r['combined_hi80']:.3f}" if has_ci else "—"))
        for _, r in top_monthly.iterrows()
    ) if not top_monthly.empty else "<tr><td colspan='4'>No active counties this month</td></tr>"

    hotspot_rows = "".join(
        f"<tr><td>{r['county_name']}, {r['state']}</td>"
        f"<td>{r['n_species']}</td><td>{r['species_list']}</td>"
        f"<td>{r['total_combined']:.3f}</td></tr>"
        for _, r in hotspots.head(10).iterrows()
    ) if not hotspots.empty else "<tr><td colspan='4'>None this period</td></tr>"

    headline = ""
    if not top_annual.empty:
        h = top_annual.iloc[0]
        headline = (f"<b>{h['county_name']}, {h['state']}</b> is the #1 forecasted "
                    f"county for {sp_short} in CY{FORECAST_YEAR} "
                    f"(annual combined risk {h['annual_combined']:.2f}).")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PESTCAST Briefing — {sp_short} · {month_name} CY{FORECAST_YEAR}</title>
<style>
  @page {{ size: letter; margin: 0.6in; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         color: #0f172a; line-height: 1.5; max-width: 900px; margin: 0 auto; padding: 24px; }}
  header {{ background: linear-gradient(90deg, #14532d, #166534); color: #ffffff;
           padding: 18px 22px; border-bottom: 3px solid #84cc16;
           border-radius: 6px; margin-bottom: 24px; }}
  header h1 {{ margin: 0; font-size: 1.4rem; letter-spacing: 0.18em;
              font-family: Georgia, serif; font-weight: 600; }}
  header .tag {{ font-size: 0.78rem; color: #bbf7d0; text-transform: uppercase;
                letter-spacing: 0.06em; margin-top: 4px; }}
  header .meta {{ font-size: 0.78rem; color: #ecfccb; margin-top: 8px; opacity: 0.9; }}
  h2 {{ font-size: 1.05rem; color: #14532d; margin: 22px 0 8px 0;
       border-bottom: 2px solid #dcfce7; padding-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 12px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; color: #475569; font-weight: 600; text-transform: uppercase;
       letter-spacing: 0.04em; font-size: 0.72rem; }}
  td:nth-child(n+3) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .headline {{ background: #fef9c3; border-left: 4px solid #ca8a04;
              padding: 10px 14px; margin: 8px 0 18px 0; border-radius: 4px; font-size: 0.95rem; }}
  .caveat {{ background: #f1f5f9; border-left: 4px solid #64748b;
            padding: 10px 14px; margin-top: 18px; font-size: 0.8rem; color: #475569; line-height: 1.5; }}
  footer {{ margin-top: 28px; font-size: 0.72rem; color: #94a3b8;
           border-top: 1px solid #e2e8f0; padding-top: 10px; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>PESTCAST</h1>
  <div class="tag">Pathway Pest Risk Intelligence — Surveillance Briefing</div>
  <div class="meta">{sp_label} · {month_name} CY{FORECAST_YEAR} · generated {now}</div>
</header>

<div class="headline">{headline}</div>

<h2>Top 10 counties — annual CY{FORECAST_YEAR} forecast</h2>
<table>
  <tr><th>County</th><th>State</th><th>Combined</th><th>Pathway μ</th><th>Climate</th></tr>
  {annual_rows}
</table>

<h2>Top 10 cells — {month_name} CY{FORECAST_YEAR}</h2>
<table>
  <tr><th>County</th><th>State</th><th>Combined</th><th>80% credible interval</th></tr>
  {monthly_rows}
</table>

<h2>Multi-species hotspots</h2>
<table>
  <tr><th>County</th><th># species in top-20</th><th>Species</th><th>Combined (sum)</th></tr>
  {hotspot_rows}
</table>

<div class="caveat">
<b>Forecast methodology.</b> Calibrated Poisson GLM fitted on APHIS PDR detection
events (CY{BASELINE_YEAR}, n=52, Pseudo-R² 0.68). County predictions distribute
state-level fitted μ proportionally by per-county pathway features
(passenger / freight / host volume from countries where the species is established
per EPPO). Climate fraction = longest cyclic run of consecutive favorable months
per WorldClim envelope. CY{FORECAST_YEAR} forecast assumes year-over-year
stability of T-100 patterns. Coefficients are illustrative; the model is
decision-support quality, not prediction-tournament quality. See PESTCAST
documentation for full methodology and caveats.
</div>

<footer>
  PESTCAST · USDA APHIS PPQ · build {VERSION} · for internal program management
</footer>
</body>
</html>
"""


def layered_risk_map(snap_active: pd.DataFrame, climate_county: pd.DataFrame,
                     centroids: pd.DataFrame, geojson: dict,
                     species_short: str) -> go.Figure:
    """
    Two-layer map:
      base    — climate suitability (faint yellow choropleth, full CONUS)
      overlay — combined-risk graduated bubbles at airport-served county centroids
    """
    base = climate_county[["fips", "county_name", "state", "long_run_mean"]].copy()
    base["fips"] = base["fips"].astype(str).str.zfill(5)
    base["frac_year_favorable"] = base["long_run_mean"].fillna(0) / 12.0

    fig = px.choropleth(
        base, geojson=geojson, locations="fips",
        color="frac_year_favorable",
        color_continuous_scale=[
            (0.0,  "#f8fafc"),  # near-white slate
            (0.25, "#fef9c3"),  # very faint yellow
            (0.5,  "#fde68a"),  # soft amber
            (0.75, "#fdba74"),  # muted orange
            (1.0,  "#f59e0b"),  # warmer orange (saturated but capped)
        ],
        range_color=(0, 1),
        scope="usa",
        labels={"frac_year_favorable": "Climate (annual)"},
    )
    fig.update_traces(
        marker_line_width=0.15,
        marker_line_color="rgba(71, 85, 105, 0.20)",
        hovertemplate="<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                      "%{z:.0%} of year favorable for establishment<extra></extra>",
        selector=dict(type="choropleth"),
    )

    # Combined-risk bubbles overlay.
    # Drop bubbles below 1% of max — they were rendering as hollow rings (very-light
    # fill + dark outline = no visible center) and adding noise instead of signal.
    bubbles = snap_active.merge(centroids, on="fips", how="inner")
    if not bubbles.empty:
        max_c = max(bubbles["combined"].max(), 0.01)
        bubbles = bubbles[bubbles["combined"] >= max_c * 0.01].copy()

    if not bubbles.empty:
        # Larger floor (8px) so small bubbles still read as filled circles, not rings.
        bubbles["size"] = np.maximum(np.sqrt(bubbles["combined"] / max_c) * 32, 8)
        # Custom red ramp that starts at a clearly-visible salmon — guarantees every
        # bubble has a saturated fill, so no more "hollow circle" optical illusion.
        red_ramp = [
            (0.0,  "#fca5a5"),  # salmon — visible at all sizes
            (0.35, "#f87171"),
            (0.65, "#dc2626"),
            (1.0,  "#7f1d1d"),  # deep red — top-tier cells
        ]
        fig.add_trace(go.Scattergeo(
            lon=bubbles["lon"],
            lat=bubbles["lat"],
            text=bubbles["county_name"].astype(str) + ", " + bubbles["state"].astype(str),
            customdata=np.stack([
                bubbles["combined"].values,
                bubbles["mu_pathway"].values,
                bubbles["frac_year_favorable"].values,
            ], axis=-1),
            mode="markers",
            marker=dict(
                size=bubbles["size"],
                color=bubbles["combined"],
                colorscale=red_ramp,
                cmin=0, cmax=max_c,
                line=dict(width=0.4, color="rgba(127, 29, 29, 0.6)"),
                sizemode="diameter",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Combined risk", font=dict(size=11, color="#475569")),
                    thickness=10, len=0.55, x=1.02, y=0.5, yanchor="middle",
                    outlinewidth=0, tickfont=dict(size=10, color="#475569"),
                ),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Combined risk: %{customdata[0]:.3f}<br>"
                "Pathway μ: %{customdata[1]:.2f}<br>"
                "Climate: %{customdata[2]:.0%}<extra></extra>"
            ),
            name="Combined risk",
            showlegend=False,
        ))

    fig.update_geos(
        showsubunits=True, subunitcolor="rgba(71, 85, 105, 0.45)", subunitwidth=0.6,
        showland=True, landcolor="#f8fafc",
        showcoastlines=True, coastlinecolor="rgba(71, 85, 105, 0.4)", coastlinewidth=0.6,
        showlakes=False, bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=dict(text="Climate (year frac)", font=dict(size=11, color="#475569")),
            thickness=10, len=0.55, x=-0.02, y=0.5, yanchor="middle", xanchor="right",
            outlinewidth=0, tickfont=dict(size=10, color="#475569"), tickformat=".0%",
        ),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif",
                  size=12, color="#0f172a"),
    )
    return fig


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-brand">
            <div class="app-brand-mark">P</div>
            <div class="app-brand-text">
                <div class="app-brand-name">PESTCAST</div>
                <div class="app-brand-tagline">Pathway Pest Risk Intelligence</div>
                <div class="app-brand-org">USDA APHIS PPQ · Surveillance Allocation</div>
            </div>
        </div>
        <div class="app-header-meta">
            <div class="app-header-horizon">Operational horizon · {HORIZON_LABEL_SHORT}</div><br/>
            <span class="app-header-meta-version">v{VERSION}</span> · session {now_str}<br/>
            <span style="opacity:0.85">{DATA_VINTAGE}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

with st.sidebar:
    # ---- Brand wordmark (top-left of sidebar) ----
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;"
        "margin:-4px 0 16px 0;padding-bottom:10px;border-bottom:1px solid #e2e8f0;'>"
        "<div style='width:30px;height:30px;border-radius:6px;"
        "background:#14532d;border:1px solid rgba(132,204,22,0.5);"
        "display:flex;align-items:center;justify-content:center;"
        "font-family:\"Iowan Old Style\",Palatino,Georgia,serif;"
        "font-weight:700;font-size:1.05rem;color:#ecfccb;'>P</div>"
        "<div style='font-family:\"Iowan Old Style\",Palatino,Georgia,serif;"
        "font-weight:600;font-size:1rem;color:#14532d;letter-spacing:0.10em;'>"
        "PESTCAST</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
        f"<span style='display:inline-block;background:#dcfce7;color:#166534;"
        f"padding:2px 8px;border-radius:999px;font-size:0.7rem;font-weight:600;"
        f"letter-spacing:0.05em;text-transform:uppercase;'>CY{FORECAST_YEAR}</span>"
        f"<span style='font-size:0.78rem;color:#64748b;'>forecast · CY{BASELINE_YEAR} features</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    pred = load_predictions()
    # Default to the first near-term month so users land on the operationally
    # reliable view, not an arbitrary 6-months-out outlook month.
    default_month = min(NEAR_TERM_MONTHS)
    month_sel = st.slider("Reporting month", 1, 12, value=default_month, format="%d")

    # 12-month visual strip showing the 90-day operational window
    strip_cells = []
    for m in range(1, 13):
        is_op  = m in NEAR_TERM_MONTHS
        is_sel = m == month_sel
        if is_op and is_sel:
            bg, fg, weight = "#15803d", "#ffffff", "700"
        elif is_op:
            bg, fg, weight = "#bbf7d0", "#14532d", "600"
        elif is_sel:
            bg, fg, weight = "#fef3c7", "#78350f", "700"
        else:
            bg, fg, weight = "#f1f5f9", "#94a3b8", "400"
        # 3-character month label (Jan, Feb...)
        strip_cells.append(
            f"<div style='flex:1;text-align:center;background:{bg};color:{fg};"
            f"font-size:0.62rem;font-weight:{weight};padding:3px 0;"
            f"border-radius:3px;letter-spacing:0.04em;'>{MONTH_NAMES[m-1][:3].upper()}</div>"
        )
    op_label = (f"{MONTH_NAMES[min(NEAR_TERM_MONTHS) - 1]}–"
                f"{MONTH_NAMES[max(NEAR_TERM_MONTHS) - 1]}")
    st.markdown(
        "<div style='margin-top:-8px;'>"
        f"<div style='display:flex;gap:2px;margin-bottom:4px;'>{''.join(strip_cells)}</div>"
        "<div style='display:flex;justify-content:space-between;"
        "font-size:0.65rem;color:#64748b;letter-spacing:0.04em;'>"
        f"<span><span style='display:inline-block;width:7px;height:7px;background:#15803d;"
        f"border-radius:2px;margin-right:4px;vertical-align:middle;'></span>"
        f"OPERATIONAL ({op_label})</span>"
        f"<span><span style='display:inline-block;width:7px;height:7px;background:#fef3c7;"
        f"border:1px solid #78350f;border-radius:2px;margin-right:4px;vertical-align:middle;'></span>"
        f"SELECTED</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    is_near = month_sel in NEAR_TERM_MONTHS
    horizon_pill_color = "#dcfce7,#166534" if is_near else "#fef3c7,#92400e"
    fg, bg = horizon_pill_color.split(",")[1], horizon_pill_color.split(",")[0]
    badge_label = "Near-term · operational" if is_near else "Outlook · planning only"
    st.markdown(
        f"<div style='margin-top:8px;'>"
        f"<span style='display:inline-block;background:{bg};color:{fg};"
        f"padding:2px 8px;border-radius:999px;font-size:0.7rem;font-weight:600;"
        f"letter-spacing:0.05em;text-transform:uppercase;'>{badge_label}</span>"
        f" <span style='font-size:0.82rem;color:#0f172a;font-weight:600;'>"
        f"{MONTH_NAMES[month_sel - 1]} {FORECAST_YEAR}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.78rem;font-weight:600;color:#475569;"
        "letter-spacing:0.04em;text-transform:uppercase;margin:14px 0 6px 0;'>"
        "Focus species</div>",
        unsafe_allow_html=True,
    )
    sp_focus = st.pills(
        "Focus species",
        options=SPECIES_LIST,
        selection_mode="single",
        default="ludens",
        format_func=lambda s: SPECIES_SHORT.get(s, s),
        key="species_pills",
        label_visibility="collapsed",
    ) or "ludens"

    # species_sel kept as a 1-element list for backward compatibility with the
    # rest of the app — every view is now single-species, so this is just sp_focus.
    species_sel = [sp_focus]

    st.caption(
        f"Drives every tab. Climate envelopes differ per species so the maps "
        f"can only show one at a time — switching here changes the entire app. "
        f"Currently viewing **{SPECIES_SHORT[sp_focus]}**."
    )

    st.markdown("---")
    st.caption(
        f"BTS T-100 publishes with a ~3–4 month lag, so the latest fully-released "
        f"calendar year is CY{BASELINE_YEAR}. The pipeline re-runs each month as "
        f"new data is released."
    )


# ---------------------------------------------------------------------------
# Headline KPIs (always visible)
# ---------------------------------------------------------------------------

cp = load_county_predictions()
mv = load_marginal()
sn = load_state_network()

# Single-species view (sp_focus drives everything via the sidebar pill).
sel = cp[cp["species"] == sp_focus].copy()
near_term = sel[sel["month"].isin(NEAR_TERM_MONTHS)]
near_term_county = (near_term.groupby(["fips", "county_name", "state"], as_index=False)
                              .agg(combined=("combined", "sum"),
                                   mu_pathway=("mu_pathway", "sum")))
top_near       = near_term_county.nlargest(1, "combined").iloc[0] if not near_term_county.empty else None
near_active    = int((near_term_county["combined"] > 0).sum())
total_pred_90d = float(near_term_county["combined"].sum())

# Top US port (90d) for this species — derived from the airport-county map and risk_table
@st.cache_data
def _top_port_for_species(species: str, baseline_year: int) -> tuple[str, float]:
    rt = load_risk_table()
    pcol = f"present_{species}"
    if pcol not in rt.columns:
        return ("—", 0.0)
    sub = rt[(rt["year"] == baseline_year) & (rt[pcol] == 1) & (rt["passengers"] > 0)]
    if sub.empty:
        return ("—", 0.0)
    by_port = sub.groupby("dest_us_port")["passengers"].sum().sort_values(ascending=False)
    return (by_port.index[0], float(by_port.iloc[0]))

top_port, top_port_pax = _top_port_for_species(sp_focus, BASELINE_YEAR)

# Annual #1 county (full year forecast)
annual_county = (sel.groupby(["fips", "county_name", "state"], as_index=False)
                    .agg(annual_combined=("combined", "sum")))
top_annual = annual_county.nlargest(1, "annual_combined").iloc[0] if not annual_county.empty else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        f"Top priority · {NEAR_TERM_LABEL}",
        f"{top_near['county_name']}, {top_near['state']}" if top_near is not None else "—",
        f"{top_near['combined']:.2f} combined risk" if top_near is not None else "",
    )
with k2:
    st.metric(
        f"Top US port · CY{BASELINE_YEAR}",
        top_port,
        f"{top_port_pax:,.0f} infested-origin pax" if top_port_pax > 0 else "",
    )
with k3:
    st.metric(
        f"Active counties · {NEAR_TERM_LABEL}",
        f"{near_active}",
        f"{total_pred_90d:.1f} expected detections (90d)" if near_active else "",
    )
with k4:
    st.metric(
        f"Annual outlook · #1 county",
        f"{top_annual['county_name']}, {top_annual['state']}" if top_annual is not None else "—",
        f"{top_annual['annual_combined']:.2f} for full CY{FORECAST_YEAR}" if top_annual is not None else "",
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pri, tab_surv, tab_path, tab_est, tab_model, tab_about = st.tabs(
    ["Priorities", "Surveillance", "Pathways", "Establishment", "Model", "About"]
)


# ---------------------------------------------------------------------------
# Tab — PRIORITIES (county risk map + actionable list)
# ---------------------------------------------------------------------------

with tab_pri:
    # Single-species view driven by the sidebar focus pill.
    sp_pri = sp_focus
    df_sp = cp[cp["species"] == sp_pri].copy()
    snap = df_sp[df_sp["month"] == month_sel].copy()
    snap_active = snap[snap["combined"] > 0].copy()

    # ------------------------------------------------------------------
    # HEADLINE RIBBON — reflects the slider's current month and tier.
    # The whole-app framing (90-day operational window vs outlook) is
    # signaled in the sidebar chip strip; here we just describe what
    # the user is currently looking at.
    # ------------------------------------------------------------------
    is_near_month = month_sel in NEAR_TERM_MONTHS
    tier_bg, tier_fg = ("#dcfce7", "#166534") if is_near_month else ("#fef3c7", "#92400e")
    tier_label_str   = "Operational" if is_near_month else "Outlook"
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:10px;margin:8px 0 14px 0;flex-wrap:wrap;'>"
        f"<span style='font-size:1rem;font-weight:600;color:#0f172a;'>"
        f"{MONTH_NAMES[month_sel - 1]} CY{FORECAST_YEAR}</span>"
        f"<span style='display:inline-block;background:{tier_bg};color:{tier_fg};"
        f"padding:2px 8px;border-radius:999px;font-size:0.68rem;font-weight:600;"
        f"letter-spacing:0.06em;text-transform:uppercase;'>{tier_label_str}</span>"
        f"<span style='font-size:0.82rem;color:#64748b;'>· {SPECIES_LABEL[sp_pri]}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    annual_sp = (df_sp.groupby(["fips", "state", "county_name"], as_index=False)
                      .agg(annual_combined=("combined", "sum"),
                           annual_pathway=("mu_pathway", "sum"),
                           climate_frac=("frac_year_favorable", "first")))
    top_max = max(annual_sp["annual_combined"].max(), 0.01)
    counties_geo = load_counties_geojson()

    if snap_active.empty:
        st.info(f"No active risk in {MONTH_NAMES[month_sel - 1]} CY{FORECAST_YEAR} "
                f"for {SPECIES_SHORT[sp_pri]}.")
    else:
        m_left, m_right = st.columns([3, 1.4])

        with m_left:
            view_mode = st.radio(
                "Map view",
                options=["layered", "risk_only", "climate_only"],
                format_func=lambda v: {
                    "layered":      "Risk + climate context",
                    "risk_only":    "Combined risk only",
                    "climate_only": "Climate suitability only",
                }[v],
                horizontal=True,
                key="pri_view",
                label_visibility="collapsed",
            )
            if view_mode == "climate_only":
                cs_county_local = load_climate_county()
                climate_sp = cs_county_local[cs_county_local["species"] == sp_pri].copy()
                climate_sp["fips"] = climate_sp["fips"].astype(str).str.zfill(5)
                climate_sp["long_run_mean"] = climate_sp["long_run_mean"].fillna(0)
                fig = professional_choropleth(
                    climate_sp, counties_geo, "long_run_mean", "Favorable months",
                    range_color=(0, 12),
                    hover_data={"county_name": True, "state": True, "long_run_mean": ":.1f"},
                    colorscale="YlOrRd",
                )
                fig.update_traces(
                    hovertemplate="<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                                  "%{z:.1f} favorable months / 12<extra></extra>"
                )
                st.plotly_chart(fig, width="stretch")
            elif view_mode == "layered":
                cs_county_local = load_climate_county()
                climate_sp = cs_county_local[cs_county_local["species"] == sp_pri].copy()
                centroids_df = load_county_centroids()
                fig = layered_risk_map(snap_active, climate_sp, centroids_df,
                                       counties_geo, SPECIES_SHORT[sp_pri])
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"**Background** = {SPECIES_SHORT[sp_pri]} climate suitability for "
                    f"establishment. **Bubbles** = combined risk in **{MONTH_NAMES[month_sel - 1]} "
                    f"CY{FORECAST_YEAR}** at airport-served counties. Move the month slider or "
                    f"switch the focus species in the **sidebar** to drive the view."
                )
            else:  # risk_only
                fig = professional_choropleth(
                    snap_active, counties_geo, "combined", "Combined risk",
                    range_color=(0, max(df_sp["combined"].max(), 0.01)),
                    hover_data={"county_name": True, "state": True,
                                "mu_pathway": ":.2f", "frac_year_favorable": ":.2f",
                                "combined": ":.3f", "fips": False},
                )
                st.plotly_chart(fig, width="stretch")

        with m_right:
            st.markdown(
                f"<div class='section-title'>Top 10 — {MONTH_NAMES[month_sel - 1]}"
                f"<span class='section-meta'>{SPECIES_SHORT[sp_pri]}</span></div>",
                unsafe_allow_html=True,
            )
            has_ci = {"combined_lo80", "combined_hi80"}.issubset(snap_active.columns)
            ci_cols = ["combined_lo80", "combined_hi80"] if has_ci else []
            t10 = snap_active.nlargest(10, "combined")[
                ["county_name", "state", "combined"] + ci_cols].copy()
            t10["Tier"] = t10["combined"].apply(lambda v: tier_label(tier(v, top_max)))
            if has_ci:
                t10["80% CI"] = t10.apply(
                    lambda r: f"{r['combined_lo80']:.2f}–{r['combined_hi80']:.2f}", axis=1)
                t10 = t10[["county_name", "state", "combined", "80% CI", "Tier"]]
                t10.columns = ["County", "State", "Risk", "80% CI", "Tier"]
            else:
                t10 = t10[["county_name", "state", "combined", "Tier"]]
                t10.columns = ["County", "State", "Risk", "Tier"]
            st.dataframe(t10, hide_index=True, width="stretch",
                         column_config={
                             "Risk": st.column_config.NumberColumn(format="%.3f"),
                         })

            # 90-day aggregate "shortcut" so users can still see operational window totals
            with st.expander(f"Show 90-day aggregate ({NEAR_TERM_LABEL})", expanded=False):
                near_sp = df_sp[df_sp["month"].isin(NEAR_TERM_MONTHS)]
                near_agg = (near_sp.groupby(["fips", "state", "county_name"], as_index=False)
                                   .agg(near_combined=("combined", "sum")))
                top_near = near_agg.nlargest(10, "near_combined")[
                    ["county_name", "state", "near_combined"]].copy()
                top_near.columns = ["County", "State", "Sum (90d)"]
                st.dataframe(top_near, hide_index=True, width="stretch",
                             column_config={
                                 "Sum (90d)": st.column_config.NumberColumn(format="%.2f"),
                             })

            csv_button(
                annual_sp.sort_values("annual_combined", ascending=False),
                f"pestcast_priorities_{sp_pri}_cy{FORECAST_YEAR}.csv", key="dl_pri_near")

    st.markdown(
        f"<div class='section-title'>Annual top 10 · 5-year trend"
        f"<span class='section-meta'>{SPECIES_SHORT[sp_pri]} · full CY{FORECAST_YEAR} forecast</span></div>",
        unsafe_allow_html=True,
    )
    a_top = annual_sp.nlargest(10, "annual_combined")[
        ["fips", "county_name", "state", "annual_combined"]].copy()
    spark = yoy_sparkline_dict(sp_pri)
    a_top["Trend (5yr)"] = a_top["fips"].map(spark)
    a_top["Tier"] = a_top["annual_combined"].apply(lambda v: tier_label(tier(v, top_max)))
    a_top = a_top[["county_name", "state", "annual_combined", "Trend (5yr)", "Tier"]]
    a_top.columns = ["County", "State", "Annual", "Trend (5yr)", "Tier"]
    st.dataframe(a_top, hide_index=True, width="stretch",
                 column_config={
                     "Annual":      st.column_config.NumberColumn(format="%.2f"),
                     "Trend (5yr)": st.column_config.LineChartColumn(
                         y_min=0, width="small",
                         help="Combined risk by year, 2020–2025"),
                 })

    # Seasonal heatmap — only counties with material risk
    THRESH_FRAC = 0.02
    heat_thresh = max(annual_sp["annual_combined"].max() * THRESH_FRAC, 0.005)
    keep = (annual_sp[annual_sp["annual_combined"] >= heat_thresh]
              .nlargest(20, "annual_combined"))

    if keep.empty:
        st.caption("Insufficient annual signal for the seasonal heatmap.")
    else:
        h = (df_sp[df_sp["fips"].isin(keep["fips"])]
                .pivot_table(index="fips", columns="month", values="combined", aggfunc="sum")
                .reindex(index=keep["fips"]))
        labels = keep["county_name"].astype(str) + ", " + keep["state"].astype(str)
        x_axis = [MONTH_NAMES[m - 1] for m in h.columns]
        fig2 = go.Figure(go.Heatmap(
            z=h.values, x=x_axis, y=labels,
            colorscale="Reds",
            colorbar=dict(title="", thickness=10, len=0.6,
                          outlinewidth=0, tickfont=dict(size=10, color="#475569")),
            hovertemplate="<b>%{y}</b><br>%{x} · combined risk %{z:.3f}<extra></extra>",
        ))

        # Fade columns past the operational horizon — visual cue that those
        # months are outlook only, not actionable forecasts.
        outlook_months = [m for m in h.columns if m not in NEAR_TERM_MONTHS]
        if outlook_months:
            outlook_indices = [list(h.columns).index(m) for m in outlook_months]
            grouped = []
            cur = [outlook_indices[0]]
            for idx in outlook_indices[1:]:
                if idx == cur[-1] + 1:
                    cur.append(idx)
                else:
                    grouped.append(cur); cur = [idx]
            grouped.append(cur)
            for grp in grouped:
                fig2.add_shape(
                    type="rect",
                    x0=grp[0] - 0.5, x1=grp[-1] + 0.5,
                    y0=-0.5, y1=len(labels) - 0.5,
                    fillcolor="rgba(248, 250, 252, 0.55)",
                    line=dict(width=0),
                    layer="above",
                )

        # Highlight the near-term band with a thin green border on top
        near_indices = [list(h.columns).index(m) for m in h.columns if m in NEAR_TERM_MONTHS]
        if near_indices:
            fig2.add_shape(
                type="rect",
                x0=min(near_indices) - 0.5, x1=max(near_indices) + 0.5,
                y0=-0.5, y1=len(labels) - 0.5,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="#15803d", width=2),
                layer="above",
            )
            fig2.add_annotation(
                x=(min(near_indices) + max(near_indices)) / 2,
                y=len(labels) - 0.15,
                text="◆ NEXT 90 DAYS · OPERATIONAL",
                showarrow=False,
                font=dict(size=10, color="#15803d", family="-apple-system, sans-serif"),
                xanchor="center", yanchor="bottom",
            )

        fig2.update_layout(
            margin=dict(l=8, r=8, t=8, b=8),
            height=max(280, 26 * len(keep) + 80),
            xaxis_title=None, yaxis_title=None,
            yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#475569")),
            xaxis=dict(tickfont=dict(size=10, color="#475569"), side="top"),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        )
        st.markdown(
            f"<div class='section-title'>Seasonality — {len(keep)} active counties"
            f"<span class='section-meta'>green box = operational window · faded = outlook</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, width="stretch")

    csv_button(annual_sp.sort_values("annual_combined", ascending=False),
               f"pestcast_priorities_{sp_pri}_cy{FORECAST_YEAR}.csv", key="dl_pri_annual")

    st.markdown("---")

    # ---- Country drivers panel ----
    drv_left, drv_right = st.columns([1, 2])
    with drv_left:
        st.markdown(
            "<div class='section-title'>What's driving this county?"
            "<span class='section-meta'>top origin contributions</span></div>",
            unsafe_allow_html=True,
        )
        active_county_options = annual_sp[annual_sp["annual_combined"] > 0].nlargest(50, "annual_combined")
        if active_county_options.empty:
            st.caption("No active counties to inspect.")
            sel_fips = None
        else:
            options = active_county_options["fips"].tolist()
            labels = (active_county_options["county_name"] + ", "
                      + active_county_options["state"]).tolist()
            label_lookup = dict(zip(options, labels))
            sel_fips = st.selectbox(
                "Inspect county",
                options=options,
                format_func=lambda f: label_lookup.get(f, f),
                key="pri_drv",
                label_visibility="collapsed",
            )
    with drv_right:
        if sel_fips:
            row = annual_sp[annual_sp["fips"] == sel_fips].iloc[0]
            st.markdown(
                f"<div style='font-size:0.95rem;font-weight:600;color:#0f172a;margin-bottom:6px;'>"
                f"{row['county_name']}, {row['state']} · "
                f"<span style='color:#475569;font-weight:400;'>annual combined risk "
                f"{row['annual_combined']:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            drivers = county_country_drivers(sp_pri, sel_fips, BASELINE_YEAR)
            if drivers.empty:
                st.caption(f"No infested-origin pathways for {SPECIES_SHORT[sp_pri]} into this county.")
            else:
                show = drivers.head(8).copy()
                show["share_pct"] = show["share"] * 100
                show = show[["origin_country", "passengers", "freight_kg", "share_pct"]]
                show.columns = ["Origin", "Passengers (CY2025)", "Freight kg (CY2025)", "Share %"]
                st.dataframe(show, hide_index=True, width="stretch",
                             column_config={
                                 "Passengers (CY2025)":  st.column_config.NumberColumn(format="%d"),
                                 "Freight kg (CY2025)":  st.column_config.NumberColumn(format="%d"),
                                 "Share %":              st.column_config.NumberColumn(format="%.1f%%"),
                             })
                top_origin = drivers.iloc[0]
                st.caption(
                    f"**{top_origin['origin_country']}** alone accounts for "
                    f"**{top_origin['share']*100:.0f}%** of the {SPECIES_SHORT[sp_pri]}-relevant "
                    f"passenger inflow into this county. A 10% drop in that route would "
                    f"reduce expected detections proportionally."
                )

    st.markdown("---")

    # ---- Multi-species hotspots ----
    st.markdown(
        "<div class='section-title'>Multi-species hotspots"
        "<span class='section-meta'>counties in top-20 for ≥2 species</span></div>",
        unsafe_allow_html=True,
    )
    hotspots = multi_species_hotspots(top_n=20)
    if hotspots.empty:
        st.caption("No counties currently in top-20 for multiple species.")
    else:
        hot_disp = hotspots.copy()
        hot_disp["species_list"] = hot_disp["species_list"].str.replace(",", " · ")
        hot_disp = hot_disp[["county_name", "state", "n_species", "species_list", "total_combined"]]
        hot_disp.columns = ["County", "State", "# species", "Species (top-20)", "Combined (sum)"]
        st.dataframe(hot_disp, hide_index=True, width="stretch",
                     column_config={
                         "# species":       st.column_config.NumberColumn(format="%d"),
                         "Combined (sum)":  st.column_config.NumberColumn(format="%.3f"),
                     })
        st.caption(
            "Counties on this list have concurrent risk for multiple fruit fly species. "
            "Inspector hours and trap deployments here cover more program priorities per dollar."
        )

    st.markdown("---")

    # ---- Briefing export ----
    bx_left, bx_right = st.columns([1, 3])
    with bx_left:
        briefing_html = _build_briefing_html(sp_pri, month_sel, annual_sp, snap_active, hotspots)
        st.download_button(
            label="📄 Generate briefing",
            data=briefing_html.encode("utf-8"),
            file_name=f"pestcast_briefing_{sp_pri}_{MONTH_NAMES[month_sel - 1].lower()}_cy{FORECAST_YEAR}.html",
            mime="text/html",
            key="dl_briefing",
            help="Download a printable HTML briefing — open in any browser and use "
                 "Cmd/Ctrl-P → Save as PDF for a leadership-ready document.",
        )
    with bx_right:
        st.caption(
            "One-click leadership briefing: cover summary, top 10 counties, multi-species "
            "hotspots, methodology footer. Print to PDF in your browser for distribution."
        )


# ---------------------------------------------------------------------------
# Tab — SURVEILLANCE (allocation)
# ---------------------------------------------------------------------------

with tab_surv:
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:10px;margin:8px 0 14px 0;'>"
        f"<span style='font-size:1.05rem;font-weight:600;color:#0f172a;letter-spacing:-0.01em;'>"
        f"Inspector-hour allocation</span>"
        f"<span style='font-size:0.78rem;color:#475569;'>·</span>"
        f"<span style='font-size:0.85rem;color:#0f172a;font-weight:600;'>{NEAR_TERM_LABEL}</span>"
        f"<span style='display:inline-block;background:#dcfce7;color:#166534;"
        f"padding:2px 8px;border-radius:999px;font-size:0.68rem;font-weight:600;"
        f"letter-spacing:0.06em;text-transform:uppercase;'>Operational window</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Allocate additional hours to the highest-marginal-value cells in the next 90 days. "
        "Diminishing returns are baked in — already-saturated cells receive less. "
        "Annual outlook is available in the expander below for budget-cycle planning."
    )

    extra_hours = st.slider(
        "Hours to deploy",
        min_value=200, max_value=20_000, value=3_000, step=200,
        key="surv_hours",
        label_visibility="collapsed",
    )
    st.caption(f"Deploying **{extra_hours:,} additional inspector-hours** in the operational window.")

    top_n = 15
    # Restrict allocation candidates to near-term months
    top = (mv[(mv["marginal_per_hour"] > 0) & (mv["month"].isin(NEAR_TERM_MONTHS))]
              .sort_values("marginal_per_hour", ascending=False)
              .head(top_n)
              .copy())
    if top.empty:
        st.warning("No positive-marginal-value cells found at current parameters.")
    else:
        weights = top["marginal_per_hour"] / top["marginal_per_hour"].sum()
        top["hours_to_add"]      = (extra_hours * weights).round().astype(int)
        top["expected_catches"]  = top["hours_to_add"] * top["marginal_per_hour"]
        top_max_mv = top["marginal_per_hour"].iloc[0]

        col1, col2 = st.columns([2.2, 1])
        with col1:
            disp = top.copy()
            disp["Cell"] = disp["state"] + " · " + disp["month"].map(lambda m: MONTH_NAMES[m - 1])
            disp["Tier"] = disp["marginal_per_hour"].apply(lambda v: tier_label(tier(v, top_max_mv)))
            disp = disp[["Cell", "Tier", "mu", "hours", "p_detect",
                         "marginal_per_hour", "hours_to_add", "expected_catches"]]
            disp.columns = ["Cell", "Tier", "Forecast μ", "Baseline hr (assumed)",
                            "p(detect)", "Marginal/hr",
                            "Add hours", "Expected catches"]
            st.dataframe(disp, hide_index=True, width="stretch",
                         column_config={
                             "Forecast μ":           st.column_config.NumberColumn(format="%.2f"),
                             "Baseline hr (assumed)":st.column_config.NumberColumn(format="%d"),
                             "p(detect)":            st.column_config.NumberColumn(format="%.2f"),
                             "Marginal/hr":          st.column_config.NumberColumn(format="%.4f"),
                             "Add hours":            st.column_config.NumberColumn(format="%d"),
                             "Expected catches":     st.column_config.NumberColumn(format="%.2f"),
                         })
            csv_button(disp, "surveillance_allocation.csv", key="dl_surv")

        with col2:
            total_catches = top["expected_catches"].sum()
            baseline_mu   = mv["mu"].sum()
            st.metric("Expected new detections", f"+{total_catches:.2f}",
                      f"+{total_catches/baseline_mu*100:.1f}% over baseline")
            n_critical = (top["marginal_per_hour"].apply(
                lambda v: tier(v, top["marginal_per_hour"].iloc[0])) == "critical").sum()
            st.metric("Critical-tier cells", f"{int(n_critical)}",
                      f"of {len(top)} positive-MV cells")
            top_cell = top.iloc[0]
            st.metric("Top allocation",
                      f"{top_cell['hours_to_add']:,} hr",
                      f"{top_cell['state']} · {MONTH_NAMES[top_cell['month'] - 1]}")

    st.markdown(
        f"<div class='outlook-section'>"
        f"<span class='outlook-eyebrow'>Annual outlook · planning context</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(
        f"Annual marginal-value rankings — full CY{FORECAST_YEAR} (planning only)",
        expanded=False,
    ):
        st.caption(
            "These rankings include cells outside the next-90-day window. "
            "Use them for budget-cycle planning, not for inspector deployment "
            "this quarter — the 12-month forecast is extrapolated under "
            "year-over-year stability and degrades past the operational horizon."
        )
        snap = mv[mv["month"] == month_sel].copy()
        if snap.empty:
            st.info("No data for the selected month.")
        else:
            snap = snap.sort_values("marginal_per_hour", ascending=False).head(20).copy()
            m_max = snap["marginal_per_hour"].max()
            snap["Tier"] = snap["marginal_per_hour"].apply(lambda v: tier_label(tier(v, m_max)))
            TIER_COLORS = {"Critical": "#b91c1c", "Elevated": "#c2410c",
                           "Watch": "#a16207", "Routine": "#16a34a"}
            fig = px.bar(
                snap, x="state", y="marginal_per_hour", color="Tier",
                color_discrete_map=TIER_COLORS,
                category_orders={"Tier": ["Critical", "Elevated", "Watch", "Routine"]},
                hover_data={"hours": ":,.0f", "p_detect": ":.3f", "mu": ":.2f", "Tier": False},
                labels={"marginal_per_hour": "Marginal Δ detections per hour", "state": "State"},
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="right", x=1, title=None,
                                          font=dict(size=11)))
            st.plotly_chart(style_chart(fig), width="stretch")


# ---------------------------------------------------------------------------
# Tab — PATHWAYS (origin-country exposure)
# ---------------------------------------------------------------------------

with tab_path:
    rt = load_risk_table()
    snap_all = rt[(rt["year"] == BASELINE_YEAR) & (rt["month"] == month_sel)].copy()

    st.markdown(
        f"<span class='info-strip'><span class='info-strip-dot'></span>"
        f"All values below are <b>observed CY{BASELINE_YEAR} actuals</b> — the "
        f"baseline driving the CY{FORECAST_YEAR} forecast.</span>",
        unsafe_allow_html=True,
    )

    # Pathways tab is filtered to the focus species' relevant origin countries
    # (countries where the species is established per EPPO).
    pcol = f"present_{sp_focus}"
    if pcol not in snap_all.columns:
        st.warning(f"No presence data for {SPECIES_SHORT[sp_focus]}.")
    else:
        snap = snap_all[(snap_all[pcol] == 1) & (snap_all["passengers"] > 0)].copy()

        st.markdown(
            f"<div class='section-title'>Top origin → US-port pathways"
            f"<span class='section-meta'>{MONTH_NAMES[month_sel - 1]} CY{BASELINE_YEAR} · "
            f"{SPECIES_SHORT[sp_focus]}-relevant origins</span></div>",
            unsafe_allow_html=True,
        )
        pathways = snap.nlargest(25, "passengers").copy()
        # Show OTHER fruit fly species also present in each origin country
        # (lets the user see whether the same pathways carry multi-species risk).
        other_species_lookup = SPECIES_LIST  # all 6
        pathways["other_species"] = pathways.apply(
            lambda r: ", ".join(SPECIES_SHORT[s] for s in other_species_lookup
                                if s != sp_focus
                                and f"present_{s}" in r.index
                                and r[f"present_{s}"] == 1),
            axis=1,
        )
        disp = pathways[["origin_country", "dest_us_port", "passengers",
                         "freight_kg", "host_kg_total", "other_species"]].copy()
        disp.columns = ["Origin", "US port", "Passengers",
                        "Freight (kg)", "Host imports (kg)",
                        f"Other species present in origin"]
        st.dataframe(disp, hide_index=True, width="stretch",
                     column_config={
                         "Passengers":         st.column_config.NumberColumn(format="%d"),
                         "Freight (kg)":       st.column_config.NumberColumn(format="%d"),
                         "Host imports (kg)":  st.column_config.NumberColumn(format="%d"),
                     })
        csv_button(disp, f"pestcast_pathways_{MONTH_NAMES[month_sel - 1].lower()}_cy{BASELINE_YEAR}.csv",
                   key="dl_path")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div class='section-title'>Top origin countries"
                f"<span class='section-meta'>{MONTH_NAMES[month_sel - 1]} CY{BASELINE_YEAR} · passengers</span></div>",
                unsafe_allow_html=True,
            )
            country_exp = (snap.groupby("origin_country", as_index=False)
                                .agg(passengers=("passengers", "sum"),
                                     ports=("dest_us_port", "nunique"),
                                     freight_kg=("freight_kg", "sum"))
                                .nlargest(15, "passengers"))
            fig = px.bar(country_exp, x="origin_country", y="passengers",
                         hover_data={"ports": True, "freight_kg": ":,.0f"},
                         labels={"origin_country": "Origin", "passengers": "Passengers"},
                         color_discrete_sequence=["#15803d"])
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,.0f} passengers<br>%{customdata[0]} US ports<br>%{customdata[1]:,.0f} kg freight<extra></extra>")
            st.plotly_chart(style_chart(fig), width="stretch")

        with c2:
            st.markdown(
                f"<div class='section-title'>Top US ports of entry"
                f"<span class='section-meta'>{MONTH_NAMES[month_sel - 1]} CY{BASELINE_YEAR} · passengers</span></div>",
                unsafe_allow_html=True,
            )
            port_exp = (snap.groupby("dest_us_port", as_index=False)
                            .agg(passengers=("passengers", "sum"),
                                 origins=("origin_country", "nunique"))
                            .nlargest(15, "passengers"))
            fig2 = px.bar(port_exp, x="dest_us_port", y="passengers",
                          hover_data={"origins": True},
                          labels={"dest_us_port": "US port (IATA)", "passengers": "Passengers"},
                          color_discrete_sequence=["#15803d"])
            fig2.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,.0f} passengers<br>from %{customdata[0]} countries<extra></extra>")
            st.plotly_chart(style_chart(fig2), width="stretch")

        st.markdown(
            f"<div class='section-title'>Annual exposure share by species"
            f"<span class='section-meta'>full-year CY{BASELINE_YEAR} · all 6 species compared</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Fraction of each species' total annual infested-origin passenger "
            f"inflow that comes from each country, CY{BASELINE_YEAR}. "
            f"This view always shows all 6 species — the focus pill in the "
            f"sidebar doesn't filter it because the chart's purpose is the "
            f"cross-species comparison."
        )
        annual = rt[rt["year"] == BASELINE_YEAR].copy()
        share_rows = []
        for s in SPECIES_LIST:  # always all 6 — the chart's job is to compare
            pcol_s = f"present_{s}"
            if pcol_s not in annual.columns:
                continue
            sub = annual[annual[pcol_s] == 1]
            tot = sub["passengers"].sum()
            top = (sub.groupby("origin_country")["passengers"].sum()
                       .div(tot if tot > 0 else 1)
                       .nlargest(8))
            for cc, pct in top.items():
                share_rows.append({"Species": SPECIES_SHORT[s], "Origin": cc, "Share": pct})
        share_df = pd.DataFrame(share_rows)
        fig3 = px.bar(share_df, x="Origin", y="Share", color="Species",
                      barmode="group",
                      labels={"Share": "Share of inflow"},
                      color_discrete_sequence=px.colors.qualitative.D3)
        fig3.update_layout(yaxis_tickformat=".0%",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                       xanchor="right", x=1, title=None,
                                       font=dict(size=11)))
        st.plotly_chart(style_chart(fig3), width="stretch")


# ---------------------------------------------------------------------------
# Tab — ESTABLISHMENT (climate suitability + scatter)
# ---------------------------------------------------------------------------

with tab_est:
    # Single-species view driven by the sidebar focus pill. Climate envelopes
    # differ per species so combining doesn't make biological sense.

    # Climate envelopes — species-specific (mirror of scripts/06_climate_suitability.py).
    CLIMATE_ENVELOPES = {
        "capitata": {"tavg_min": 16, "tavg_max": 32, "prec_min": 20,
                     "note": "Mediterranean — broad subtropical envelope"},
        "dorsalis": {"tavg_min": 13, "tavg_max": 32, "prec_min": 50,
                     "note": "Wet tropical — needs higher precipitation"},
        "ludens":   {"tavg_min": 14, "tavg_max": 30, "prec_min": 30,
                     "note": "Subtropical — moderate temp ceiling"},
        "suspensa": {"tavg_min": 16, "tavg_max": 32, "prec_min": 30,
                     "note": "Caribbean — humid subtropical"},
        "zonata":   {"tavg_min": 14, "tavg_max": 32, "prec_min": 40,
                     "note": "Tropical Asia/Africa"},
        "cerasi":   {"tavg_min":  8, "tavg_max": 25, "prec_min": 30,
                     "note": "TEMPERATE — inverted from the others; thrives in cool climates"},
    }

    cs_county = load_climate_county()
    counties_geo = load_counties_geojson()

    env = CLIMATE_ENVELOPES[sp_focus]
    scope_label = SPECIES_SHORT[sp_focus]

    e_left, e_right = st.columns([3, 1.5])
    with e_left:
        st.markdown(
            f"<div class='section-title'>County climate suitability"
            f"<span class='section-meta'>{SPECIES_SHORT[sp_focus]} · "
            f"tavg {env['tavg_min']}–{env['tavg_max']}°C · "
            f"prec ≥{env['prec_min']} mm/mo</span></div>",
            unsafe_allow_html=True,
        )
        county_sp = cs_county[cs_county["species"] == sp_focus].copy()
        county_sp["fips"] = county_sp["fips"].astype(str).str.zfill(5)
        county_sp["long_run_mean"] = county_sp["long_run_mean"].fillna(0)
        fig = professional_choropleth(
            county_sp, counties_geo, "long_run_mean", "Favorable months",
            range_color=(0, 12),
            hover_data={"county_name": True, "state": True, "long_run_mean": ":.1f"},
            colorscale="YlOrRd",
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                          "%{z:.1f} favorable months / 12<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"**{env['note']}.** Color shows the longest cyclic run of consecutive "
            f"favorable months at this species' envelope. **Each species has its "
            f"own envelope** — switch the pill above to see how the map shifts. "
            f"Cherry fly (cerasi) is the most striking case: temperate species, "
            f"so cooler northern counties light up while tropical ones go cold. "
            f"12 = year-round suitable."
        )

    with e_right:
        # Combined risk table — single species (the one chosen in the pill above the map).
        st.markdown(
            f"<div class='section-title'>Top 20 counties — combined risk"
            f"<span class='section-meta'>{scope_label}</span></div>",
            unsafe_allow_html=True,
        )
        annual_county_est = (cp[cp["species"] == sp_focus]
                                .groupby(["fips", "state", "county_name"], as_index=False)
                                .agg(pathway_mu_year=("mu_pathway", "sum"),
                                     combined_year=("combined", "sum"),
                                     climate_frac=("frac_year_favorable", "first")))
        show = annual_county_est.nlargest(20, "combined_year")[
            ["county_name", "state", "combined_year", "pathway_mu_year", "climate_frac"]
        ].copy()
        show.columns = ["County", "State", "Combined", "Pathway", "Climate"]
        st.dataframe(show, hide_index=True, width="stretch", height=620,
                     column_config={
                         "Combined": st.column_config.NumberColumn(format="%.3f"),
                         "Pathway":  st.column_config.NumberColumn(format="%.2f"),
                         "Climate":  st.column_config.NumberColumn(format="%.2f"),
                     })
        csv_button(show, f"pestcast_establishment_{sp_focus}.csv", key="dl_est_top")

    st.markdown(
        f"<div class='section-title'>Pathway × climate map"
        f"<span class='section-meta'>{scope_label} · annual</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each point is one airport-served county. **Top-right** = priority surveillance. "
        "**Top-left** = pathway-rich but climate-blocked (irrigation-gap candidates). "
        "**Bottom-right** = vulnerable to establishment if a detection occurs upstream."
    )
    sc = (cp[cp["species"] == sp_focus]
              .groupby(["fips", "state", "county_name"], as_index=False)
              .agg(pathway_mu_year=("mu_pathway", "sum"),
                   combined_year=("combined", "sum"),
                   climate_frac=("frac_year_favorable", "first")))
    sc = sc[sc["pathway_mu_year"] > 0].copy()
    sc["label"] = sc["county_name"] + ", " + sc["state"]

    if sc.empty:
        st.info("No airport-served counties with positive pathway risk for this species.")
    else:
        fig_sc = px.scatter(
            sc, x="pathway_mu_year", y="climate_frac",
            size="combined_year", color="combined_year",
            color_continuous_scale="Reds",
            hover_name="label",
            hover_data={"pathway_mu_year": ":.2f", "climate_frac": ":.2f",
                        "combined_year": ":.3f", "label": False, "fips": False,
                        "state": False, "county_name": False},
            labels={"pathway_mu_year": "Pathway μ (annual predicted detections)",
                    "climate_frac": "Climate suitability (fraction of year favorable)",
                    "combined_year": "Combined"},
        )
        top6 = sc.nlargest(6, "combined_year").reset_index(drop=True)
        # Spread the annotations directionally so they don't overlap
        offsets = [(20, -22), (-20, -22), (20, 22), (-20, 22), (30, 0), (-30, 0)]
        for i, (_, r) in enumerate(top6.iterrows()):
            ax_, ay_ = offsets[i % len(offsets)]
            fig_sc.add_annotation(x=r["pathway_mu_year"], y=r["climate_frac"],
                                  text=r["label"], showarrow=True, arrowhead=1,
                                  ax=ax_, ay=ay_, font=dict(size=10, color="#334155"))
        fig_sc.update_layout(height=520)
        st.plotly_chart(style_chart(fig_sc), width="stretch")


# ---------------------------------------------------------------------------
# Tab — MODEL (diagnostics)
# ---------------------------------------------------------------------------

with tab_model:
    st.markdown(
        "<div class='section-title'>Model fit & residuals"
        f"<span class='section-meta'>Poisson GLM · fitted on CY{BASELINE_YEAR} · 52 detection events · Pseudo-R² 0.68</span></div>",
        unsafe_allow_html=True,
    )

    bd = load_breakdown().copy()
    bd["resid"] = bd["detections"] - bd["mu"]
    bd["abs_resid"] = bd["resid"].abs()
    biggest = bd.loc[bd["resid"].idxmax()]

    h1, h2 = st.columns([1, 2.4])
    with h1:
        st.metric(
            "Largest under-prediction",
            f"{biggest['state']} · {MONTH_NAMES[int(biggest['month']) - 1]}",
            f"{int(biggest['detections'])} observed · {biggest['mu']:.2f} predicted · "
            f"{SPECIES_SHORT[biggest['species']]}",
        )
    with h2:
        st.markdown(
            f"<div style='background:#fef2f2; border:1px solid #fecaca; "
            f"border-radius:8px; padding:12px 16px; font-size:0.88rem; color:#7f1d1d; line-height:1.5;'>"
            f"<b>What this means.</b> {SPECIES_LABEL[biggest['species']]} in "
            f"{biggest['state']} ({MONTH_NAMES[int(biggest['month']) - 1]} {BASELINE_YEAR}) "
            f"saw <b>{int(biggest['detections'])} detections</b> against the model's "
            f"<b>{biggest['mu']:.2f} predicted</b>. The pathway features don't "
            "fully explain the surge — likely an unmodeled driver such as a "
            "specific outbreak cluster, an unobserved arrival route, or an "
            "irrigation-driven establishment dynamic. Worth investigating before v2."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-title'>Coefficients"
        "<span class='section-meta'>Poisson GLM · log link</span></div>",
        unsafe_allow_html=True,
    )
    coefs = load_coefficients().copy()
    coefs["exp_coef"] = np.exp(coefs["coef"])
    coefs["sig"] = np.where(coefs["p_value"] < 0.001, "***",
                    np.where(coefs["p_value"] < 0.01, "**",
                    np.where(coefs["p_value"] < 0.05, "*", "")))
    disp = coefs[["term", "coef", "exp_coef", "std_err", "p_value", "sig"]].copy()
    disp.columns = ["Term", "β", "exp(β)", "Std err", "p-value", ""]
    st.dataframe(disp, hide_index=True, width="stretch",
                 column_config={
                     "β":        st.column_config.NumberColumn(format="%.3f"),
                     "exp(β)":   st.column_config.NumberColumn(format="%.3f"),
                     "Std err":  st.column_config.NumberColumn(format="%.3f"),
                     "p-value":  st.column_config.NumberColumn(format="%.4f"),
                 })
    csv_button(disp, "model_coefficients.csv", key="dl_coefs")
    st.caption("On the log scale: β means a one-unit increase in the feature multiplies expected detections by exp(β). "
               "Significance: *** p<0.001 · ** p<0.01 · * p<0.05.")

    st.markdown(
        "<div class='section-title'>Residuals"
        "<span class='section-meta'>state × month × species · top 15 by absolute residual</span></div>",
        unsafe_allow_html=True,
    )
    surprises = bd.nlargest(15, "abs_resid")[
        ["state", "month", "species", "detections", "mu", "resid", "p_detect", "marginal_per_hour"]
    ].copy()
    surprises["month"] = surprises["month"].map(lambda m: MONTH_NAMES[m - 1])
    surprises["species"] = surprises["species"].map(SPECIES_SHORT)
    surprises.columns = ["State", "Month", "Species", "Observed",
                         "Predicted μ", "Residual", "p(detect)", "Marginal/hr"]
    st.dataframe(surprises, hide_index=True, width="stretch",
                 column_config={
                     "Predicted μ":  st.column_config.NumberColumn(format="%.2f"),
                     "Residual":     st.column_config.NumberColumn(format="%.2f"),
                     "p(detect)":    st.column_config.NumberColumn(format="%.3f"),
                     "Marginal/hr":  st.column_config.NumberColumn(format="%.4f"),
                 })
    st.caption("Positive residual: more detections than predicted (unrecognized driver). "
               "Negative residual: predicted detections did not materialize "
               "(effective surveillance, or under-reporting in validation labels).")

    st.markdown(
        f"<div class='section-title'>Network structure"
        f"<span class='section-meta'>top 15 states by inbound passengers · Oct CY{BASELINE_YEAR}</span></div>",
        unsafe_allow_html=True,
    )
    snap_n = sn[(sn["year"] == BASELINE_YEAR) & (sn["month"] == 10)].copy()
    snap_n = snap_n.nlargest(15, "state_total_pax")
    share_cols = [f"state_inf_share_{s}" for s in SPECIES_LIST if f"state_inf_share_{s}" in snap_n.columns]
    cols = ["state", "state_n_countries", "state_origin_entropy", "state_total_pax"] + share_cols
    show = snap_n[cols].copy()
    rename = {"state": "State", "state_n_countries": "# origin countries",
              "state_origin_entropy": "Origin entropy", "state_total_pax": "Total inbound pax"}
    for c in share_cols:
        sp = c.replace("state_inf_share_", "")
        rename[c] = f"{SPECIES_SHORT[sp]} share" if sp in SPECIES_SHORT else f"{sp} share"
    show = show.rename(columns=rename)
    cfg = {"Origin entropy":    st.column_config.NumberColumn(format="%.2f"),
           "Total inbound pax": st.column_config.NumberColumn(format="%d")}
    for c, name in rename.items():
        if "share" in name:
            cfg[name] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(show, hide_index=True, width="stretch", column_config=cfg)


# ---------------------------------------------------------------------------
# Tab — ABOUT (methodology + sources)
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown("### About PESTCAST")
    st.markdown(
        "PESTCAST predicts fruit-fly detection risk from foreign air pathways and "
        "recommends where additional inspector-hours catch the most pests, at "
        "county granularity, for the contiguous United States. Built for "
        "USDA APHIS PPQ surveillance program managers.\n\n"
        "*A portmanteau of \"pest\" and \"forecast\" — like a weather-cast for "
        "national-scale agricultural pest risk.*"
    )

    with st.expander(f"Forecast horizon — CY{FORECAST_YEAR}", expanded=True):
        st.markdown(
            f"**Producing a CY{FORECAST_YEAR} calendar-year forecast** using "
            f"**CY{BASELINE_YEAR}** monthly features (T-100 international passenger / "
            f"freight, USDA FAS GATS host-commodity imports, EPPO species presence, "
            f"WorldClim climate normals)."
        )
        st.markdown(
            f"##### Why CY{BASELINE_YEAR} features"
            f"\nBTS publishes T-100 with a typical 3–4 month lag; USDA FAS GATS "
            f"lags by ~2 months. As of session start, the latest fully-released year "
            f"is CY{BASELINE_YEAR}. Year-over-year travel and trade patterns are "
            f"highly stable outside macro shocks (COVID 2020–2021 being the obvious "
            f"exception), so prior-year actuals are the standard operational forecast "
            f"baseline used by airlines, freight planners, and BTS itself."
        )
        st.markdown(
            f"##### CY{FORECAST_YEAR}-specific data already incorporated"
            f"\n- USDA FAS GATS January–February {FORECAST_YEAR} (released)\n"
            f"- IPPC pest reports through April {FORECAST_YEAR}\n"
            f"- APHIS PDR detection events through April {FORECAST_YEAR} (14 events) — "
            f"these are *back-test labels*; the model is fit on CY{BASELINE_YEAR} "
            f"actuals, and CY{FORECAST_YEAR} detections are future events the "
            f"forecast is being tested against."
        )
        st.markdown(
            "##### Refresh cadence\n"
            "Re-run the pipeline each month as BTS and FAS publish new monthly "
            "extracts. The forecast updates automatically — no model retraining "
            "required for routine months."
        )

    with st.expander("Methodology", expanded=False):
        st.markdown("##### Pathway exposure")
        st.markdown(
            "Bilateral monthly inbound flight segments (BTS T-100) are filtered "
            "to international arrivals, joined with EPPO species-presence records "
            "to identify infested-origin pathways, and aggregated to "
            "(origin country × US airport × month). USDA FAS GATS provides "
            "per-country host-commodity import volumes for seven HS-4 host "
            "categories (citrus, stone fruit, mangoes, melons, berries/grapes/kiwi, "
            "tomatoes, cucumbers)."
        )
        st.markdown("##### Risk model")
        st.markdown(
            "A Poisson GLM is fitted at the (state × month × species) grain — the "
            "level at which APHIS PDR detection labels exist. Features: log-volume "
            "of infested passengers / freight / host imports; Fourier seasonality "
            "(annual + semi-annual); origin-country diversity (Shannon entropy and "
            "species-specific infested share). Pseudo-R² ≈ 0.68 on 52 validation "
            "events spanning 2018–2026."
        )
        st.markdown("##### County views")
        st.markdown(
            "Fitted state coefficients are applied to per-county pathway features "
            "(airport→county spatial join), then rescaled within each "
            "(state, month, species) so county μ's sum to the calibrated state μ. "
            "This preserves the labels-fit total while distributing risk across "
            "counties using their own arrival profiles."
        )
        st.markdown("##### Establishment")
        st.markdown(
            "WorldClim 10-arcmin monthly normals are used to compute the longest "
            "cyclic run of consecutive favorable months per raster pixel for each "
            "species' temperature/precipitation envelope, then aggregated to "
            "counties via zonal statistics. **Combined risk = pathway μ × "
            "(climate fraction)**."
        )
        st.markdown("##### Surveillance allocation")
        st.markdown(
            "An exponential-capture detection model (p_detect = 1 − exp(−h/k)) "
            "recovers latent arrival rates from fitted μ and current pro-rata "
            "staffing. Marginal value of an additional hour is "
            "(A / k)(1 − p_detect), encoding diminishing returns. The deployment "
            "slider distributes a hypothetical block of additional hours across "
            "top cells proportional to marginal value."
        )

    with st.expander("Data sources", expanded=False):
        sources = [
            ("BTS T-100 International Segment", "DOT BTS TranStats", "Inbound air passenger and freight by route, monthly", "2015–2025 · 11 years"),
            ("USDA FAS GATS", "USDA FAS", "Imported fruit/vegetable host commodities by partner country, monthly", "2015–2026 · 12 years"),
            ("EPPO Global Database", "European/Mediterranean PPO", "Country-level pest presence per species", "2024-12-10 export"),
            ("IPPC pest reports", "International Plant Protection Convention", "Recent quarantine actions and detections", "2025–2026"),
            ("APHIS PDR", "USDA APHIS Plant Protection", "Federal-order detection events used as model labels", "Through 2026-04"),
            ("WorldClim v2.1", "WorldClim.org", "Global monthly climate normals (10-arcmin)", "Static reference"),
            ("USDA NASS Cropland Data Layer", "USDA NASS", "U.S. crop-class raster (CDL 2025, 30 m)", "2025 — v2 candidate, currently unused"),
            ("Natural Earth", "naturalearthdata.com", "Country boundaries (1:50m)", "Static reference"),
            ("US Census TIGER", "U.S. Census Bureau", "County boundaries with FIPS codes", "Static reference"),
            ("OurAirports", "ourairports.com", "Airport metadata (IATA, ICAO, lat/lon)", "Continuous"),
        ]
        src_df = pd.DataFrame(sources, columns=["Dataset", "Provider", "Description", "Coverage"])
        st.dataframe(src_df, hide_index=True, width="stretch")

    with st.expander("Species coverage (6 modeled)", expanded=False):
        species_table = pd.DataFrame([
            {"Species": SPECIES_LABEL["capitata"], "EPPO Code": "CERTCA",
             "Climate envelope": "tavg 16–32 °C · prec ≥20 mm",
             "Validation events": 20},
            {"Species": SPECIES_LABEL["dorsalis"], "EPPO Code": "DACUDO",
             "Climate envelope": "tavg 13–32 °C · prec ≥50 mm",
             "Validation events": 14},
            {"Species": SPECIES_LABEL["ludens"], "EPPO Code": "ANSTLU",
             "Climate envelope": "tavg 14–30 °C · prec ≥30 mm",
             "Validation events": 31},
            {"Species": SPECIES_LABEL["suspensa"], "EPPO Code": "ANSTSU",
             "Climate envelope": "tavg 16–32 °C · prec ≥30 mm",
             "Validation events": 2},
            {"Species": SPECIES_LABEL["zonata"], "EPPO Code": "DACUZO",
             "Climate envelope": "tavg 14–32 °C · prec ≥40 mm",
             "Validation events": 2},
            {"Species": SPECIES_LABEL["cerasi"], "EPPO Code": "RHAGCE",
             "Climate envelope": "tavg 8–25 °C · prec ≥30 mm (temperate)",
             "Validation events": 5},
        ])
        st.dataframe(species_table, hide_index=True, width="stretch")

    with st.expander("Caveats and limitations", expanded=False):
        st.markdown("##### T-100 is bilateral")
        st.markdown(
            "Connecting passengers (e.g. Vietnam → Korea → LAX) are attributed "
            "to the last foreign segment, not the original boarding country. "
            "Network-structure features partially compensate by characterizing "
            "each port's origin-country diversity."
        )
        st.markdown("##### Irrigation gap")
        st.markdown(
            "WorldClim uses *natural* monthly precipitation. Irrigated agricultural "
            "regions (notably California's Central Valley) sustain fly populations "
            "even when natural precipitation falls below the species minimum, so "
            "climate scores there understate true establishment risk. A v2 layer "
            "would composite the climate envelope with a USDA NASS CDL irrigated-"
            "host mask."
        )
        st.markdown("##### Validation sample size")
        st.markdown(
            "52 detection events across 6 species. Coefficients are illustrative "
            "more than definitive; sign and rough magnitude are reliable, exact "
            "effect sizes have wide bands."
        )
        st.markdown("##### Counties without airports")
        st.markdown(
            "Pathway = 0 even though they receive spillover from neighboring "
            "airport counties. A v2 layer would diffuse arrivals via county-"
            "adjacency or commute matrices."
        )
        st.markdown("##### Network features inherited from state")
        st.markdown(
            "Small counties with one airport carry the parent state's origin-"
            "country mix, which can be misleading for outlier ports."
        )

    with st.expander("Roadmap", expanded=False):
        st.markdown(
            "- Composite irrigation × climate envelope using NASS CDL host-crop mask\n"
            "- Seasonal phenology coding per species (peak-month weighting)\n"
            "- Bayesian hierarchical model with credible intervals replacing the GLM\n"
            "- Connecting-flight attribution via the BTS DB1B 10% itinerary sample\n"
            "- Ingest CDFA / Florida DACS detection bulletins to expand validation\n"
            "- Maritime cargo (LA, Miami, NY/NJ container imports) as additional pathway"
        )

    st.markdown("---")
    st.caption(f"PESTCAST · build {VERSION} · {DATA_VINTAGE} · "
               "© USDA APHIS PPQ — for internal program management.")
