import xarray as xr
import rioxarray
import geopandas as gpd
from pathlib import Path

# Source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

src_dir = Path("../dataset/nclimgrid/")
dst_dir = Path("../dataset/croppednclimgrid/")
dst_dir.mkdir(exist_ok=True, parents=True)

us = gpd.read_file("../dataset/censusshape/tl_2025_us_state.shp")
ca = us[us['NAME'] == 'California'].to_crs("EPSG:4326")

for nc_file in src_dir.glob("*.nc"):
    print(f"Cropping → {nc_file.name}")
    ds = xr.open_dataset(nc_file, chunks={"time": 365})
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds_ca = ds.rio.clip(ca.geometry, ca.crs)
    out_name = f"ca-{nc_file.name}"
    out_path = dst_dir / out_name
    ds_ca.to_netcdf(out_path)

