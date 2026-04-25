# Fruit Fly Risk — Data Acquisition Plan

## Goal
Assemble a minimum viable dataset to score, by U.S. port-of-entry × month, the risk of fruit fly introduction from foreign pathways, and validate against historical interception patterns where possible.

## Scope decisions for the hackathon
- **Target species**: Mediterranean fruit fly (*Ceratitis capitata*), Oriental fruit fly (*Bactrocera dorsalis*), Mexican fruit fly (*Anastrepha ludens*). These three account for the bulk of high-consequence detections in the contiguous U.S.
- **Pathways**: International air passengers and air cargo first (most tractable from public data). Maritime and land-border cargo as stretch goals.
- **Time window**: Most recent 3 full calendar years of monthly data. Enough to see seasonality without drowning the team.
- **Geography**: Foreign country of origin → U.S. airport (port of entry). Aggregate to U.S. county/state for the establishment-side analysis.

## The five datasets we actually need

### 1. Pest presence by country (and ideally by month)
**What it is**: Which countries have established populations of each target species, ideally with seasonal abundance.

**Source**:
- *CABI Compendium* datasheets — country-level presence/absence lists exist in the public datasheet pages (e.g., Ceratitis capitata datasheet 12367). Full distribution tables are gated.
- *EPPO Global Database* (gd.eppo.int) — open, has country-level presence records per species.
- *Best shortcut*: the 2014 PLOS One paper "Global Assessment of Seasonal Potential Distribution of Mediterranean Fruit Fly" — supplementary materials contain occurrence points with monthly data. Same authors have similar work for *B. dorsalis*.

**Action**: Scrape EPPO Global Database for each target species → country-level presence. Manually code seasonal severity (high/medium/low) per country-month from the published phenology literature.

### 2. Foreign → U.S. air passenger volumes by route, monthly
**What it is**: How many people flew from origin airport X (in country Y) to U.S. airport Z in month M.

**Source**: BTS TranStats T-100 International Segment.
- URL: https://www.transtats.bts.gov/Tables.asp?QO_VQ=EEE
- Table: "T-100 International Segment (All Carriers)"
- Filter: arrival country = USA, all foreign departure countries.
- Fields needed: YEAR, MONTH, ORIGIN (foreign airport ICAO/IATA), ORIGIN_COUNTRY, DEST (US airport), PASSENGERS, FREIGHT, MAIL.
- Format: CSV download via the Prezipped File option, ~50–200 MB per year.

**Action**: Download last 3 calendar years. One CSV per year. Filter to international segments only.

### 3. Foreign → U.S. host commodity imports, monthly
**What it is**: Volume (kg) of fresh fruit fly host commodities imported from each country each month, ideally by port of entry.

**Source**:
- *Primary*: USDA FAS GATS (apps.fas.usda.gov/gats) — Standard Query → FAS U.S. Trade → Imports → select HTS codes for host commodities → by partner country, monthly.
- *Alternative*: Census Bureau USA Trade Online (usatrade.census.gov) — same underlying data, different UI, free but requires account.

**Host HTS codes (the short list)**:
- 0805: Citrus (oranges, mandarins, grapefruit, lemons, limes)
- 0804.50: Guavas, mangoes, mangosteens
- 0809: Apricots, cherries, peaches, plums
- 0807: Melons, papayas
- 0810: Other fresh fruit (strawberries, grapes, etc.)
- 0702/0707: Tomatoes, cucumbers (Medfly hosts)

**Action**: Pull monthly import quantity (kg) and value (USD) by partner country for each HTS code. Sum into a single "host commodity volume" per country-month. Port-of-entry breakdown is harder — Census has it but only at the district level, not always cleanly mapped to an airport.

### 4. APHIS pest interceptions (validation target)
**What it is**: Historical record of where and when pests were actually intercepted at U.S. ports.

**Source**: APHIS AQAS — *not openly downloadable*. Workarounds:
- Annual APHIS PPQ Program Data Reports (aphis.usda.gov/wildlife-services/publications/pdr) — aggregated counts.
- Published academic papers using this data — the PLOS One 2014 Medfly paper is the gold standard reference. Supplementary tables list interceptions by year/state.
- FOIA request (not realistic in hackathon timeframe).
- *Practical fallback*: hand-code a small validation set from published outbreak/detection news (CDFA Mediterranean fruit fly project records for California are public; Florida DACS publishes detection bulletins).

**Action**: Don't block the project on this. Build the model with what we can get, and use the published aggregates from the 2014 paper or APHIS annual reports as a coarse validation check.

### 5. U.S. climate suitability and host crop locations
**What it is**: Where in the U.S. would each target species actually be able to establish if it arrived?

**Sources**:
- *Climate*: WorldClim v2.1 monthly normals (worldclim.org/data/worldclim21.html) — free, gridded, global. Or PRISM (prism.oregonstate.edu) for U.S.-only at higher resolution.
- *Host crops*: USDA NASS Cropland Data Layer (nassgeodata.gmu.edu/CropScape) — annual U.S. crop raster, ~30m resolution. Filter to fruit/vegetable host crops.
- *Existing climate suitability rasters*: The 2014 PLOS One paper produced and published a global Medfly suitability raster — checking the supplementary materials before re-running MaxEnt is the right call.

**Action**: Use existing published suitability rasters where possible. For each U.S. county, compute (a) climate suitability score (0–1) per month, (b) acreage of host crops within county. Multiply for an "establishment vulnerability" index.

## Data we are explicitly NOT chasing
- DB1B (10% ticket sample, complex itineraries) — too messy for a hackathon, T-100 segment data is cleaner.
- TSA passenger screening — not origin-specific, low signal.
- BTS Freight Analysis Framework — modeled estimates, not direct measurements; redundant with T-100 air freight.
- UN Comtrade — duplicates GATS for U.S.-side data.
- Land border crossings — important in reality (Mexico → Texas/California is huge for *A. ludens*) but adds another data source. Stretch goal.

## Suggested ingestion order (first day)
1. **Get T-100 first** (1–2 hours). It's the largest file, and downloading it sets the temporal frame.
2. **Pest presence countries** (1 hour). Manual list of ~30–60 countries per species from EPPO.
3. **GATS host commodity imports** (2–3 hours). Tedious UI but valuable.
4. **Climate + host suitability** (2 hours). Use pre-existing rasters where possible.
5. **APHIS validation set** (parallel, opportunistic). Whatever we can scrape from public reports.

## Output format for downstream modeling
Standardize everything to one canonical join key: **(origin_country, dest_us_airport_or_county, year, month)**. Every dataset gets reshaped to that grain. Final modeling table has one row per country-port-month with columns for passenger volume, cargo volume, pest presence flag, seasonal severity, U.S. climate suitability, U.S. host acreage near port.

## File layout
```
data/
  raw/                          # untouched downloads
    t100_international_2023.csv
    t100_international_2024.csv
    ...
    eppo_pest_presence.csv      # hand-built from EPPO
    gats_host_imports.csv       # exported from GATS
    worldclim_us_monthly.tif
    cdl_2024.tif
  processed/
    pathway_volumes.parquet     # T-100 cleaned and filtered
    origin_risk.parquet         # pest presence × seasonal severity per country-month
    establishment_risk.parquet  # per US county-month
    risk_table.parquet          # final joined table for the app
```
