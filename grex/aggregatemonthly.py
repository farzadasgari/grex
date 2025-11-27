import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

src_dir = Path("../dataset/croppednclimgrid/")
monthly_list = []

for nc_file in sorted(src_dir.glob("ca-*.nc")):
    parts = nc_file.name.split('-')
    if len(parts) < 3:
        continue
    yyyymm = parts[2][:6]
    if not yyyymm.isdigit() or len(yyyymm) != 6:
        continue
    year = yyyymm[:4]
    month = yyyymm[4:6]
    year_month = f"{year}.{month}"
    print(f"Processing {year_month}")
    ds_daily = xr.open_dataset(nc_file, chunks={'time': 31})

    if 'tavg' in ds_daily:
        tmean_daily = ds_daily['tavg']
    else:
        tmean_daily = (ds_daily['tmax'] + ds_daily['tmin']) / 2

    precip_monthly = ds_daily['prcp'].sum(dim='time')
    tmean_monthly = tmean_daily.mean(dim='time')

    monthly_slice = xr.Dataset({
        'prcp': precip_monthly,
        'tmean': tmean_monthly
    })
    monthly_slice = monthly_slice.expand_dims(time=[pd.to_datetime(f"{year}-{month}-01")])
    monthly_list.append(monthly_slice)

monthly_ds = xr.concat(monthly_list, dim='time')
monthly_ds = monthly_ds.sortby('time').dropna(dim='time', how='all')
monthly_ds.to_netcdf("../dataset/monthly_ca_1951_2025.nc")