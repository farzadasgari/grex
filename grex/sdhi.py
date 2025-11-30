import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm
from tqdm import tqdm

x_da = xr.open_dataarray("../dataset/x_ca_1951_2025.nc")

warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")
mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)

years = x_da.year.values
n_years = len(years)

sdhi_arr = np.full((n_years, len(x_da.lat), len(x_da.lon)), np.nan)

for c_idx in tqdm(range(n_cells), desc="Processing cells"):
    sel_lat_idx = lat_idxs[c_idx]
    sel_lon_idx = lon_idxs[c_idx]
    cell_x = x_da.isel(lat=sel_lat_idx, lon=sel_lon_idx).values
    valid_mask = ~np.isnan(cell_x)
    if np.sum(valid_mask) < 10: continue
    x_valid = cell_x[valid_mask]
    n_valid = len(x_valid)
    rank_x = rankdata(x_valid, method='average')
    f_x = (rank_x - 0.44) / (n_valid + 0.12)
    f_x = np.clip(f_x, 1e-6, 1 - 1e-6)
    sdhi_valid = norm.ppf(f_x)
    sdhi_arr[valid_mask, sel_lat_idx, sel_lon_idx] = sdhi_valid

sdhi_da = xr.DataArray(
    sdhi_arr,
    dims=['year', 'lat', 'lon'],
    coords={
        'year': years,
        'lat': x_da.lat,
        'lon': x_da.lon
    },
    name='sdhi'
)

sdhi_da.attrs['description'] = 'Standardized Dry and Hot Index (SDHI) per year per cell (warm season)'
sdhi_da.to_netcdf("../dataset/sdhi_ca_1951_2025.nc")
