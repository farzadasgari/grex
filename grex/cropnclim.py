import xarray as xr
import rioxarray
import geopandas as gpd

# Source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

us = gpd.read_file("../dataset/censusshape/tl_2025_us_state.shp")
