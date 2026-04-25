# Fruit Fly Pathway Risk Dashboard — USDA APHIS PPQ
CSU Hackathon 2026, Prompt 1. Two interactive front ends for fruit fly introduction risk into the contiguous U.S. via foreign air passenger and cargo pathways — a static geospatial dashboard and a full analytical intelligence platform. Six target species: *Ceratitis capitata* (Medfly), *Bactrocera dorsalis* (Oriental FF), *Anastrepha ludens* (Mexican FF), *Anastrepha suspensa* (Caribbean FF), *Bactrocera zonata* (Peach FF), and *Rhagoletis cerasi* (European Cherry FF).

## Two front ends

| | Leaflet Dashboard (`app/index.html`) | PESTCAST (`app/app.py`) |
|---|---|---|
| **Stack** | Static HTML + Leaflet.js + Chart.js | Streamlit + Plotly + scikit-learn |
| **Data** | Pre-built JS bundle (`app_data.js`) | Live Parquet files in `data/processed/` |
| **Species** | 3 (Medfly, Oriental FF, Mexican FF) | 6 (all target species) |
| **Risk model** | Deterministic composite index | Poisson GLM (Pseudo-R² 0.68) |
| **Forecasting** | Monthly index, current year | CY2026 forecast + 90-day operational window |
| **Best for** | Quick geospatial overview, presentations | Operational analysis, resource allocation |

## Pipeline at a glance

```
BTS T-100 (2015–2025) ──┐
USDA FAS GATS imports ──┼──→ 01_acquire_data.py ──→ 02_build_join_table.py ──→ risk_table.parquet
EPPO pest presence ──────┤
APHIS interceptions ─────┘         │
                                   ├──→ 03_fit_risk_model.py   ──→ predictions.parquet
                                   ├──→ 04_marginal_value.py   ──→ marginal_value.parquet
                                   ├──→ 05_network_features.py ──→ network features
                                   ├──→ 05_build_app_data.py   ──→ app/data/app_data.js (Leaflet bundle)
                                   └──→ 06_climate_suitability.py ──→ climate_suitability_by_county.parquet
                                              │
WorldClim tavg/prec ──┐            07_county_predict.py ──→ county_predictions.parquet
USDA CDL (2025) ──────┘            08_backtest.py / 09_surveillance_backtest.py

                                   ▼                          ▼
                         app/index.html                  app/app.py
                      (Leaflet choropleth +          (PESTCAST Streamlit —
                       flight arcs + ports)           6-tab analytics platform)
```

Risk score = `(passengers + freight / 5 000) × pest_presence_score`, aggregated monthly per `(origin_country × dest_port × species)`. Pest presence scored from EPPO distribution records: 3 = widespread, 2 = restricted, 1 = few occurrences / transient.

## Data sources (verified)

| Source | Use | Years | Access |
|---|---|---|---|
| BTS T-100 International Segment | Passenger + freight volumes by route | 2015–2025 | BTS TranStats download |
| USDA FAS GATS | Host commodity imports (kg) by partner country | 2015–2026 | GATS Standard Query |
| EPPO Global Database | Pest presence status per country, 6 target species | current | gd.eppo.int distribution export |
| APHIS PPQ Program Data | Interception / detection validation records | 2015–2026 | APHIS public bulletins |
| IPPC Pest Reports | Phytosanitary event feed (2025–2026) | 2025–2026 | IPPC REST API |
| OurAirports | Airport metadata — IATA, lat/lon | current | davidmegginson.github.io |
| Natural Earth 50m | Country boundaries GeoJSON | current | nvkelso/natural-earth-vector |
| WorldClim v2.1 | Monthly avg temperature + precipitation rasters | 1970–2000 normals | worldclim.org |
| USDA NASS CDL | U.S. host crop acreage mask (2025 clip) | 2025 | CropScape / GEE |
| ISO 3166-1 country codes | Country name ↔ ISO2 reconciliation | current | datasets/country-codes |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API keys required for either dashboard. T-100 and GATS data require manual export from BTS TranStats and USDA FAS GATS portals respectively (see [`DATA_PLAN.md`](DATA_PLAN.md) for step-by-step instructions).

## Data pipeline (run once)

```bash
# 1. Check which raw data files are present
python scripts/01_acquire_data.py --check

# 2. Auto-download files that can be fetched directly (airports, country codes, GeoJSON)
python scripts/01_acquire_data.py --auto

# 3. Build the join table from all raw CSVs
python scripts/02_build_join_table.py

# 4. Fit the Poisson GLM risk model
python scripts/03_fit_risk_model.py

# 5. Compute marginal surveillance value per county
python scripts/04_marginal_value.py

# 6. Compute climate suitability by county (WorldClim + CDL)
python scripts/06_climate_suitability.py

# 7. Generate county-level risk predictions
python scripts/07_county_predict.py

# 8. Build the Leaflet app data bundle (app/data/app_data.js)
python scripts/05_build_app_data.py
```

`05_build_app_data.py` is idempotent — re-running it regenerates the bundle from whatever raw files are present. Expected runtime: ~15 seconds for the full dataset.

## Running the front ends

### Option A — Leaflet.js geospatial dashboard (no dependencies after build)

```bash
cd app && python3 -m http.server 8765
# open http://localhost:8765
```

Serves `index.html` — works entirely in the browser with no server process.

### Option B — PESTCAST Streamlit analytics platform

```bash
.venv/bin/streamlit run app/app.py
# opens automatically in browser, default http://localhost:8501
```

Requires the processed Parquet files in `data/processed/` (generated by the pipeline above).

## Dashboard features

### Leaflet dashboard (`index.html`)

**Map layers (all toggleable)**
- Country choropleth in two modes: *Pest Presence* (EPPO score categories) or *Pathway Risk* (gradient heatmap — flight volume × pest score)
- Great-circle flight arcs for top 15 / 30 / 50 risk routes, colored and weighted by risk tier
- U.S. port-of-entry bubbles sized by total inbound risk score; click for full origin breakdown

**Controls**
- Month slider (Jan–Dec) with play/pause animation — all layers and charts update live
- Species filter: All Species / Medfly / Oriental FF / Mexican FF
- Include/exclude cargo from risk calculation
- Port IATA labels overlay
- Focus CONUS / World view buttons

**Sidebar panels** (each collapsible; sidebar itself collapsible)
- Risk summary stats: origin country count, total passengers, cargo tonnage, ports exposed
- Top 12 risk pathways ranked with species badges and bar chart
- U.S. port exposure: top 10 ports with stacked species-mix chart and APHIS state detection grid
- Monthly passenger flow trend (Chart.js line, all 3 species overlaid)
- Host commodity imports by partner country (GATS, horizontal bar)
- Recent detections & IPPC alerts feed (combined, sorted by date)
- Risk model methodology callout

### PESTCAST Streamlit app (`app/app.py`) — 6 tabs

**Sidebar** — species focus pill (6 species), month selector, cargo toggle, data vintage stamp

**Priorities tab** — operational command view for the selected species and month
- Combined pathway × climate risk map (county-level bubbles over climate suitability choropleth)
- Top 10 high-risk counties with operational status badges (High / Medium / Watch)
- 90-day aggregate view for the near-term operational window
- Annual top-10 county trend with 5-year sparklines
- County seasonality heatmap with active-window overlay
- County driver breakdown — top origin country contributions per county
- Multi-species hotspot table (counties ranking top-20 for ≥2 species simultaneously)
- Printable HTML leadership briefing export

**Surveillance tab** — inspector-hour allocation optimizer
- Marginal-value model with diminishing-returns baked in
- Hour deployment slider (200–20,000 hrs) allocated across the highest-value county × month cells
- 90-day operational window focus; annual outlook expander for budget planning

**Pathways tab** — origin → U.S. port route analysis
- Top origin → US-port pathways filtered to EPPO-established countries for the focus species
- Monthly top origin countries and top U.S. ports of entry
- Annual exposure share by species (all 6 compared, not filtered)

**Establishment tab** — climate suitability at the county level
- Species-specific climate envelope (WorldClim tavg / prec thresholds)
- County choropleth: fraction of year with favorable establishment conditions
- Pathway risk vs. climate suitability scatter — identifies counties with both high arrival pressure and high establishment potential

**Model tab** — Poisson GLM diagnostics
- Model fit summary: Pseudo-R² 0.68, fitted on 52 CY2025 detection events
- Largest under-prediction callout with plain-language interpretation
- Coefficient table and residual plot
- State-level network structure visualization

**About tab** — data sources, methodology, data vintage, and version info

## Project layout

```
scripts/
  01_acquire_data.py          auto-download + manual instructions for every source
  02_build_join_table.py      merge T-100, GATS, EPPO, APHIS → risk_table.parquet
  03_fit_risk_model.py        Poisson GLM on APHIS detection records
  04_marginal_value.py        marginal surveillance value per county
  05_build_app_data.py        process raw CSVs → app/data/app_data.js (Leaflet bundle)
  05_network_features.py      airport-county network graph features
  06_climate_suitability.py   WorldClim + CDL → climate_suitability_by_county.parquet
  07_county_predict.py        county-level CY2026 risk predictions
  08_backtest.py              model backtesting
  09_surveillance_backtest.py surveillance allocation backtesting
data/
  raw/                        one file per source — gitignored
    t100_international_*.csv  BTS T-100, 2015–2025
    eppo_*.csv                EPPO distribution records, 6 species
    gats_host_imports.csv
    aphis_validation.csv
    ippc_fruit_fly_pest_reports_2025_2026.csv
    airports.csv
    countries.geojson
    wc2.1_10m_tavg_*.tif      WorldClim temperature normals (12 months)
    wc2.1_10m_prec_*.tif      WorldClim precipitation normals (12 months)
    cdl_2025_clipped.tif      USDA CDL host crop mask
  processed/                  generated by pipeline — consumed by app/app.py
    risk_table.parquet
    predictions.parquet
    marginal_value.parquet
    climate_suitability_by_county.parquet
    county_predictions.parquet
app/
  index.html                  single-file Leaflet.js + Chart.js dashboard (no server needed)
  app.py                      PESTCAST Streamlit analytics platform
  data/
    app_data.js               generated JS bundle (pest presence + routes + GeoJSON)
DATA_PLAN.md                  full data acquisition plan and source rationale
```

## Design notes

**Two complementary tools.** The Leaflet dashboard is a zero-dependency presentation layer — shareable without a Python environment. The Streamlit app is the full analytical platform: Poisson GLM forecasts, marginal surveillance value, county-level establishment risk, and leadership briefings.

**Risk score is pathway-weighted, not model-predicted (Leaflet).** The static dashboard computes a deterministic composite index — `volume × pest_score` — producing interpretable, auditable scores. The Streamlit app layers a fitted Poisson GLM on top for detection-calibrated forecasts.

**Cargo scaling.** T-100 FREIGHT is in pounds; passengers are integer headcounts. Raw freight values are ~5,000× larger than passenger values. The `freight / 5 000` scaling factor approximately equalises their contributions. The "Include Cargo Risk" toggle lets analysts compare passenger-only vs. combined pathway risk.

**Choropleth opacity encodes pathway volume.** Countries with the same EPPO pest score are distinguished by opacity — higher-volume routes render darker, making Mexico visually dominant over a low-traffic widespread-presence country like Ghana.

**Pre-computation at build time.** `05_build_app_data.py` pre-aggregates risk by `(species × month × country)` and `(species × month × US_port)`. The browser performs no heavy aggregation at runtime — all slider and filter updates are O(n_countries) lookups.

**90-day operational window.** The PESTCAST app flags the next ~90 days as the operationally reliable forecast horizon. Month cells outside this window are rendered faded and labeled as "outlook" to distinguish near-term actionable risk from longer-range projections.

**Hardware.** Data processing and raster clipping (WorldClim, CDL) run locally. The DGX Spark (128 GB unified memory, Blackwell architecture, BF16) was used for computationally intensive raster operations and to accelerate the full 11-year T-100 corpus pipeline.

**Validation.** Predicted high-risk corridors (Mexico → TX/CA for *A. ludens*; Thailand / Kenya → West Coast for *B. dorsalis* / *C. capitata*) align with APHIS 2025–2026 detection bulletins and IPPC phytosanitary reports included in the dataset. Model Pseudo-R² of 0.68 on 52 held-out detection events.
