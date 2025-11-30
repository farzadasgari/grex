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