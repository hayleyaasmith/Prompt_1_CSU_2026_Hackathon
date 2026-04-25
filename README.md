# Fruit Fly Risk — CSU Hackathon

Geospatial risk model and interactive map for fruit fly introduction into the
contiguous U.S. via foreign air passenger and cargo pathways.

## Quick start

```bash
# 1. See what data we need and the status of each file
python scripts/01_acquire_data.py --check

# 2. Print full instructions (auto downloads + manual steps for everything else)
python scripts/01_acquire_data.py
```

## Repo layout

```
docs/
  DATA_PLAN.md           # Full data acquisition plan and rationale
data/
  raw/                   # Untouched downloads — gitignore this
  processed/             # Cleaned, joined, ready for modeling
scripts/
  01_acquire_data.py     # Download + manual instructions
  02_clean_t100.py       # (todo) Filter and reshape T-100 to canonical grain
  03_join_pathway.py     # (todo) Join pathway, pest, climate into one risk table
  04_score_risk.py       # (todo) Compute composite risk score per port × month
app/
  app.py                 # (todo) Streamlit/Dash interactive map
```

## Pipeline at a glance

1. **Pathway** = (passengers + host cargo volume) from country C to U.S. port P in month M.
   Source: BTS T-100 international segment + USDA FAS GATS imports.
2. **Origin risk** = pest presence in country C × seasonal abundance × pathway volume.
   Source: EPPO Global Database for presence; published phenology for seasonality.
3. **Establishment risk** = U.S. climate suitability × host crop acreage near port P.
   Source: WorldClim climate normals; USDA NASS Cropland Data Layer.
4. **Composite risk** = origin × establishment, indexed by (port, month).
5. **Validation** = correlate predicted risk with APHIS interception aggregates where
   available (limited — see DATA_PLAN.md).

## Open questions for the team

- Should we include maritime cargo? Seaports handle huge fruit volumes (Port of LA,
  Miami, NY/NJ) and could be a major risk pathway not covered by T-100.
- Land border crossings (especially Mexico → TX/CA for *Anastrepha ludens*) —
  add as v2 if time permits.
- How will we present uncertainty? Risk scores from this kind of model can be off
  by 2× or more. Probably show a tertile (low/med/high) rather than a continuous
  number on the map.
