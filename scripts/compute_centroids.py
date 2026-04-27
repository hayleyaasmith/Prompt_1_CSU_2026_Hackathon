import geopandas as gpd
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Load and compute centroids
counties = gpd.read_file(RAW / "geo" / "us_counties.geojson")
counties["fips"] = (counties["STATE"].astype(str).str.zfill(2)
                    + counties["COUNTY"].astype(str).str.zfill(3))
proj = counties.to_crs(epsg=5070)  # CONUS Albers equal-area
centroids = proj.geometry.centroid.to_crs(epsg=4326)

# Save as parquet
df = pd.DataFrame({
    "fips": counties["fips"].values,
    "lon":  centroids.x.values,
    "lat":  centroids.y.values,
})
df.to_parquet(PROCESSED / "county_centroids.parquet", index=False)
print(f"Saved {len(df)} county centroids to county_centroids.parquet")
