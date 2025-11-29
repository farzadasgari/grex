import xarray as xr
import numpy as np

monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)

if len(lat_idxs) == 0:
    raise "No valid cells found — check file."

rnd_idx = np.random.choice(len(lat_idxs))
selected_lat = warm_da.lat[lat_idxs[rnd_idx]].values
selected_lon = warm_da.lon[lon_idxs[rnd_idx]].values

cell_warm = warm_da.sel(lat=selected_lat, lon=selected_lon)
window_months = cell_warm.values.astype(int)
print(f"Random cell at lat={selected_lat:.2f}, lon={selected_lon:.2f}")
print(f"Warm window: {window_months} (months {window_months[0]}-{window_months[1]}-{window_months[2]})")

cell_p = monthly_ds['prcp'].sel(lat=selected_lat, lon=selected_lon, method='nearest')
cell_t = monthly_ds['tmean'].sel(lat=selected_lat, lon=selected_lon, method='nearest')
years = np.unique(cell_p['time.year'].values)

warm_p = np.full(len(years), np.nan)
warm_t = np.full(len(years), np.nan)
for y_idx, year in enumerate(years):
    year_data_p = cell_p.sel(time=str(year))
    year_data_t = cell_t.sel(time=str(year))
    if len(year_data_p.time) < 12: continue
    month_mask = year_data_p['time.month'].isin(window_months)
    if month_mask.sum() == 3:
        warm_p[y_idx] = year_data_p.where(month_mask, drop=True).mean('time').values
        warm_t[y_idx] = year_data_t.where(month_mask, drop=True).mean('time').values

valid_mask = ~np.isnan(warm_p)
warm_p = warm_p[valid_mask]
warm_t = warm_t[valid_mask]
years = years[valid_mask]
n = len(years)

if n < 10:
    raise "Too few valid years — rerun for new random cell."