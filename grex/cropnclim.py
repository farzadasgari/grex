import xarray as xr
import rioxarray
import geopandas as gpd

# Source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

us = gpd.read_file("../dataset/censusshape/tl_2025_us_state.shp")
ca = us[us['NAME'] == 'California']
ca = ca.to_crs("EPSG:4326")

ds = xr.open_dataset("../dataset/nclimgrid/ncdd-195101-grd-scaled.nc")
ds = ds.rio.write_crs("EPSG:4326")
ds_ca = ds.rio.clip(ca.geometry, ca.crs)
ds_ca.to_netcdf("tmax_CA.nc")
