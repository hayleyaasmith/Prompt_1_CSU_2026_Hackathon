"""
Fruit Fly Risk — Data acquisition script

Run this from the project root. It does what it can automatically and
prints clear step-by-step instructions for the data sources that require
manual download (most of them, frankly — federal data portals are a maze).

Usage:
    python scripts/01_acquire_data.py            # print all instructions
    python scripts/01_acquire_data.py --auto     # run automatic downloads only
    python scripts/01_acquire_data.py --check    # report what's already downloaded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Automatic downloads: things we can grab with a plain HTTP GET
# ---------------------------------------------------------------------------

AUTO_SOURCES = [
    {
        "name": "OurAirports — airport metadata (IATA, ICAO, country, lat/lon)",
        "url": "https://davidmegginson.github.io/ourairports-data/airports.csv",
        "subdir": "geo",
        "filename": "airports.csv",
        "notes": "Used to map T-100 origin airport codes to country of origin.",
    },
    {
        "name": "Country code lookup (ISO 3166-1 alpha-2 / alpha-3 / numeric)",
        "url": "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv",
        "subdir": "geo",
        "filename": "country_codes.csv",
        "notes": "Used to reconcile country names across BTS, GATS, and EPPO data.",
    },
    {
        "name": "Natural Earth — country boundaries (1:50m, GeoJSON)",
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
        "subdir": "geo",
        "filename": "countries.geojson",
        "notes": "For map rendering — country choropleths of pest presence and risk.",
    },
    {
        "name": "US county boundaries (Census TIGER, simplified GeoJSON)",
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        "subdir": "geo",
        "filename": "us_counties.geojson",
        "notes": "For map rendering — county-level establishment risk choropleth.",
    },
]


def download(url: str, dest: Path) -> tuple[bool, str]:
    """Returns (success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"already present ({dest.stat().st_size:,} bytes)"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fruitfly-hackathon/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True, f"downloaded ({len(data):,} bytes)"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def run_auto() -> None:
    print("\n=== Automatic downloads ===\n")
    for src in AUTO_SOURCES:
        dest = RAW / src["subdir"] / src["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, msg = download(src["url"], dest)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {src['name']}")
        print(f"      → {dest.relative_to(ROOT)}  [{msg}]")
        print(f"      {src['notes']}\n")


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------

MANUAL_SOURCES = [
    {
        "name": "BTS T-100 International Segment (passenger + freight, monthly)",
        "subdir": "trade",
        "expected_files": ["t100_international_2023.csv", "t100_international_2024.csv", "t100_international_2025.csv"],
        "instructions": dedent("""
            1. Open https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4%20Pn44vr45
               (TranStats → Aviation → Air Carrier Statistics (T-100) → International Segment (All Carriers))

            2. In the field selector, check at minimum:
                  YEAR, MONTH,
                  PASSENGERS, FREIGHT, MAIL,
                  CARRIER, ORIGIN, ORIGIN_COUNTRY, ORIGIN_COUNTRY_NAME,
                  DEST, DEST_COUNTRY, DEST_COUNTRY_NAME

            3. Set the geography filter:
                  Origin Country: ALL
                  Destination Country: United States (US)
               (We only want flights *into* the U.S.)

            4. Pick a year (the form only returns one year at a time). Click "Download".
               Repeat for the last 3 calendar years.

            5. Each download is a ZIP containing one CSV. Unzip and rename to:
                  data/raw/trade/t100_international_<YEAR>.csv

            File size: ~30–80 MB unzipped per year.
        """).strip(),
    },
    {
        "name": "USDA FAS GATS — fresh fruit imports by country, monthly",
        "subdir": "trade",
        "expected_files": ["gats_host_imports.csv"],
        "instructions": dedent("""
            1. Open https://apps.fas.usda.gov/gats/  → click "Standard Query".

            2. Settings:
                  Data Source: FAS U.S. Trade
                  Product Type: Imports
                  Time Period: Monthly, last 36 months

            3. Products → Search by HTS code, add ALL of these:
                  0805    Citrus fruit (oranges, mandarins, grapefruit, lemons, limes)
                  080440  Avocados
                  080450  Guavas, mangoes, mangosteens
                  0807    Melons (incl. watermelons), papayas
                  0809    Apricots, cherries, peaches, plums, sloes
                  0810    Other fresh fruit (berries, kiwi, etc.)
                  0702    Tomatoes, fresh or chilled
                  0707    Cucumbers and gherkins, fresh

            4. Partners → Select "All countries" (or top 50 by value if you want a smaller file).

            5. Statistics → Quantity (kg). Optionally also pull Value (USD).

            6. Click "Run Query" → "Export to Excel" or "Export to CSV".
               Save as data/raw/trade/gats_host_imports.csv

            Note: GATS reports by aggregated U.S. customs district, not individual ports.
                  We'll join districts to airport groups in the cleaning step.
        """).strip(),
    },
    {
        "name": "EPPO Global Database — pest presence by country (per target species)",
        "subdir": "pests",
        "expected_files": ["eppo_ceratitis_capitata.csv", "eppo_bactrocera_dorsalis.csv", "eppo_anastrepha_ludens.csv"],
        "instructions": dedent("""
            For each of the three target species:

            1. Open https://gd.eppo.int and search for the species:
                  - Ceratitis capitata           (EPPO code: CERTCA)  — Mediterranean fruit fly
                  - Bactrocera dorsalis          (EPPO code: DACUDO)  — Oriental fruit fly
                  - Anastrepha ludens            (EPPO code: ANSTLU)  — Mexican fruit fly

            2. On the species page, click the "Distribution" tab.

            3. Use the "Export" button (top right) → CSV.
               Each export is one country per row with status (Present / Absent / Eradicated /
               Transient). Save to:
                  data/raw/pests/eppo_ceratitis_capitata.csv
                  data/raw/pests/eppo_bactrocera_dorsalis.csv
                  data/raw/pests/eppo_anastrepha_ludens.csv

            4. Manual seasonal coding: EPPO doesn't give per-month presence. For each present
               country, look up the published phenology (peak abundance months) from:
                  - The 2014 PLOS One paper (Szyniszewska & Tatem) — supplementary materials
                    have Medfly seasonality by region.
                  - CABI Compendium datasheets for the species (free preview shows seasonality
                    notes in the Biology section).
               Code as: peak_months = [3, 4, 5, ...]  for the lookup table.
        """).strip(),
    },
    {
        "name": "WorldClim — global monthly climate normals",
        "subdir": "climate",
        "expected_files": [f"wc2.1_10m_{var}_{m:02d}.tif" for var in ("tavg", "prec") for m in range(1, 13)],
        "instructions": dedent("""
            1. Open https://worldclim.org/data/worldclim21.html

            2. Download "Monthly weather data" at 10-minute resolution:
                  - Average temperature (tavg)   ~13 MB
                  - Precipitation (prec)         ~13 MB
               (10-minute resolution is plenty for county-level analysis. Higher res is
               available but balloons file sizes.)

            3. Unzip into data/raw/climate/. You should end up with 24 GeoTIFFs:
                  wc2.1_10m_tavg_01.tif … wc2.1_10m_tavg_12.tif
                  wc2.1_10m_prec_01.tif … wc2.1_10m_prec_12.tif
               The cleaning step will clip these to the contiguous U.S.

            We use this to compute climate suitability: which U.S. counties are warm
            enough year-round to allow fruit fly establishment?
        """).strip(),
    },
    {
        "name": "USDA NASS Cropland Data Layer — U.S. host crop locations",
        "subdir": "landcover",
        "expected_files": ["cdl_2025_clipped.tif"],
        "instructions": dedent("""
            1. Open https://nassgeodata.gmu.edu/CropScape/

            2. Use the AOI tool to draw a bounding box around the contiguous U.S.
               (or skip and download the full national raster — it's ~1.5 GB compressed).

            3. Download the most recent year (2025 as of writing). Save as:
                  data/raw/landcover/cdl_2025_clipped.tif

            4. We'll filter to host crop classes during processing:
                  204 (Citrus), 211 (Olives), 217 (Stone Fruit), 218 (Peaches), 220 (Plums),
                  221 (Strawberries), 222 (Squash), 223 (Apricots), 226 (Oranges),
                  227 (Lettuce), 242 (Blueberries), 246 (Radishes), 250 (Cranberries),
                  68 (Apples), 69 (Grapes), 76 (Walnuts), 77 (Pears), 212 (Oranges),
                  213 (Honeydew), 214 (Cherries), 216 (Peppers).

            Optional shortcut: USDA NASS publishes county-level acreage tables in CSV
            (https://quickstats.nass.usda.gov) — much smaller than the raster and
            sufficient if you don't need sub-county precision.
        """).strip(),
    },
    {
        "name": "APHIS pest interception aggregates (validation, not training)",
        "subdir": "pests",
        "expected_files": ["aphis_validation.csv"],
        "instructions": dedent("""
            The full AQAS interception database is INTERNAL to USDA. We can't get
            row-level interception data from public sources in a hackathon timeframe.

            Workarounds for validation:

            1. APHIS PPQ Annual Data Reports:
                  https://www.aphis.usda.gov/wildlife-services/publications/pdr
               Aggregated counts of pest detections per fiscal year, sometimes broken
               out by port/state. Hand-extract relevant tables.

            2. Published outbreak records (state-level):
                  - California CDFA fruit fly project: https://www.cdfa.ca.gov/plant/pdep/
                  - Florida DACS-DPI: https://www.fdacs.gov/Divisions-Offices/Plant-Industry
                  Both publish historical Mediterranean / Oriental / Mexican fruit fly
                  detection records.

            3. The Szyniszewska & Tatem 2014 PLOS One paper:
                  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0111582
               Supplementary tables list interception data per port for Medfly. This
               was funded by USDA-APHIS specifically for risk modeling, so the
               aggregates are public.

            4. Save your hand-compiled validation set as a small CSV:
                  data/raw/pests/aphis_validation.csv
               Columns: year, month, state_or_port, species, count
        """).strip(),
    },
]


def show_manual() -> None:
    print("\n=== Manual download steps ===\n")
    for i, src in enumerate(MANUAL_SOURCES, start=1):
        print(f"{i}. {src['name']}")
        print(f"   Expected file(s): {', '.join(src['expected_files'])}")
        print()
        for line in src["instructions"].splitlines():
            print(f"   {line}")
        print()


def check_status() -> None:
    print("\n=== Status check ===\n")
    all_files: list[tuple[str, str, str]] = []
    for src in AUTO_SOURCES:
        all_files.append((src["subdir"], src["filename"], "auto"))
    for src in MANUAL_SOURCES:
        for f in src["expected_files"]:
            all_files.append((src["subdir"], f, "manual"))
    for subdir, fname, kind in all_files:
        path = RAW / subdir / fname
        rel = f"{subdir}/{fname}"
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ [{kind}]  {rel}  ({size:,} bytes)")
        else:
            print(f"  ✗ [{kind}]  {rel}  (missing)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire fruit fly risk datasets.")
    parser.add_argument("--auto", action="store_true", help="Run automatic downloads only.")
    parser.add_argument("--check", action="store_true", help="Report what's already in data/raw/{geo,pests,trade,climate,landcover}.")
    args = parser.parse_args()

    if args.check:
        check_status()
        return 0

    if args.auto:
        run_auto()
        return 0

    # Default: show everything.
    run_auto()
    show_manual()
    check_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
