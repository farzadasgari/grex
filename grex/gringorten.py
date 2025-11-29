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