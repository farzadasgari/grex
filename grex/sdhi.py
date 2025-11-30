import xarray as xr
import numpy as np

x_da = xr.open_dataarray("../dataset/x_ca_1951_2025.nc")

warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")
mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)

years = x_da.year.values
n_years = len(years)

sdhi_arr = np.full((n_years, len(x_da.lat), len(x_da.lon)), np.nan)
