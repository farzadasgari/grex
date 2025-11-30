import xarray as xr
import numpy as np

ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
tmean = ds['tmean']
climatology = tmean.groupby('time.month').mean('time')
annual_range = climatology.max('month') - climatology.min('month')
land_mask = annual_range > 9.0
max_temp = np.full(climatology.shape[1:], -999.0)
warm_season = np.full((*climatology.shape[1:], 3), np.nan)
for start_month in range(12):
    months = np.array([(start_month + j) % 12 + 1 for j in range(3)])
    window_mean = climatology.sel(month=months.tolist()).mean('month')
    candidate = window_mean.values
    valid = ~np.isnan(candidate)
    better = (candidate > max_temp) & land_mask & valid

    if better.any():
        max_temp[better] = candidate[better]
        warm_season[better, 0] = months[0]
        warm_season[better, 1] = months[1]
        warm_season[better, 2] = months[2]
warm_da = xr.DataArray(
    warm_season,
    dims=['lat', 'lon', 'window'],
    coords={
        'lat': climatology.lat,
        'lon': climatology.lon,
        'window': [1, 2, 3]
    },
    name='warm_season_months'
).where(land_mask)

warm_da.attrs['description'] = ('Three consecutive months with the highest mean temperature '
                                '(climatological warm season). Ocean masked out.')
warm_da.attrs['land_mask_threshold'] = 'annual_range > 9.0°C'
warm_da.to_netcdf("../dataset/warm_seasons_ca.nc")

print("\nMost common warm seasons over land:")
unique, counts = np.unique(warm_season[land_mask.values], axis=0, return_counts=True)
for months, count in zip(unique[np.argsort(-counts)], np.sort(counts)[::-1]):
    if not np.isnan(months).any():
        print(f"  {int(months[0]):02d}-{int(months[1]):02d}-{int(months[2]):02d} : {count} cells")
        # 06 - 07 - 08: 11672 cells → Jun–Jul–Aug
        # 07 - 08 - 09: 11373 cells → Jul–Aug–Sep
