import xarray as xr
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)

n_cells = len(lat_idxs)
if n_cells == 0:
    raise ValueError("No valid cells found — check file.")

years = np.unique(monthly_ds['time.year'].values)
n_years = len(years)

x_arr = np.full((n_years, len(monthly_ds.lat), len(monthly_ds.lon)), np.nan)

for c_idx in tqdm(range(n_cells), desc="Processing cells"):
    sel_lat = warm_da.lat[lat_idxs[c_idx]].values
    sel_lon = warm_da.lon[lon_idxs[c_idx]].values

    cell_warm = warm_da.sel(lat=sel_lat, lon=sel_lon)
    window_months = cell_warm.values.astype(int)

    cell_p = monthly_ds['prcp'].sel(lat=sel_lat, lon=sel_lon, method='nearest')
    cell_t = monthly_ds['tmean'].sel(lat=sel_lat, lon=sel_lon, method='nearest')

    warm_p = np.full(n_years, np.nan)
    warm_t = np.full(n_years, np.nan)
    for y_idx, year in enumerate(years):
        year_data_p = cell_p.sel(time=cell_p['time.year'] == year)
        year_data_t = cell_t.sel(time=cell_t['time.year'] == year)
        month_values = year_data_p['time.month'].values
        month_mask = np.isin(month_values, window_months)
        if np.sum(month_mask) == 3:
            warm_p[y_idx] = year_data_p.values[month_mask].mean()
            warm_t[y_idx] = year_data_t.values[month_mask].mean()

    valid_mask = ~np.isnan(warm_p)
    if np.sum(valid_mask) < 10:
        continue

    warm_p_valid = warm_p[valid_mask]
    warm_t_valid = warm_t[valid_mask]
    years_valid = years[valid_mask]
    n_valid = len(years_valid)

    rank_p = rankdata(warm_p_valid, method='average')
    rank_t = rankdata(warm_t_valid, method='average')

    g1 = (rank_p - 0.44) / (n_valid + 0.12)
    g2 = (rank_t - 0.44) / (n_valid + 0.12)

    eps = 1e-6
    g1 = np.clip(g1, eps, 1 - eps)
    g2 = np.clip(g2, eps, 1 - eps)

    x_valid = g1 / g2

    for v_idx, year in enumerate(years_valid):
        full_y_idx = np.where(years == year)[0][0]
        x_arr[full_y_idx, lat_idxs[c_idx], lon_idxs[c_idx]] = x_valid[v_idx]

x_da = xr.DataArray(
    x_arr,
    dims=['year', 'lat', 'lon'],
    coords={
        'year': years,
        'lat': monthly_ds.lat,
        'lon': monthly_ds.lon
    },
    name='x'
)
x_da.attrs['description'] = 'Compound dry-hot ratio X = G1(P)/G2(T) per year per cell (warm season)'
x_da.to_netcdf("../dataset/x_ca_1951_2025.nc")
