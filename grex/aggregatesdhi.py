import xarray as xr
import numpy as np
import pandas as pd

sdhi_da = xr.open_dataarray("../dataset/sdhi_ca_1951_2025.nc")
thresholds = {
    'abnormal': -0.5,
    'moderate': -0.8,
    'severe': -1.3,
    'extreme': -1.6,
    'exceptional': -2.0
}

land_mask = ~np.isnan(sdhi_da.mean(dim='year'))
total_land_cells = land_mask.sum().item()

metrics = []
for year in sdhi_da.year.values:
    year_sdhi = sdhi_da.sel(year=year)
    valid_year_mask = ~np.isnan(year_sdhi)
    year_land_cells = valid_year_mask.sum().item()
    row = {'year': year, 'mean_sdhi': year_sdhi.mean().values}
    for level, thresh in thresholds.items():
        below = (year_sdhi <= thresh) & valid_year_mask
        pct = (below.sum().item() / year_land_cells) * 100 if year_land_cells > 0 else np.nan
        row[f'pct_{level}'] = pct
    metrics.append(row)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("../dataset/sdhi_ca_metrics.csv", index=False)
