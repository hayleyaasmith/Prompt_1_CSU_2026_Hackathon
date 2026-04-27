import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.app import professional_choropleth

PROCESSED = Path('data/processed')
RAW = Path('data/raw')

with open(RAW / 'geo' / 'us_counties.geojson', 'r', encoding='utf-8') as f:
    geo = json.load(f)

df = pd.read_parquet(PROCESSED / 'risk_table.parquet')
df = df[df['year'] == 2025].copy()
df['fips'] = df['fips'].astype(str).str.zfill(5)
print('county_name exists:', 'county_name' in df.columns)
print('sample cols:', list(df.columns)[:20])

fig = professional_choropleth(df.head(5), geo, 'combined', 'Combined risk', range_color=(0,1), hover_data={})
fig.update_traces(customdata=df.head(5)[['county_name', 'state']].values,
                  hovertemplate='<b>%{customdata[0]}, %{customdata[1]}</b><br>%{z:.1f}<extra></extra>')
print('trace count:', len(fig.data))
print('customdata shape:', fig.data[0].customdata.shape)
print('hovertemplate:', fig.data[0].hovertemplate)
