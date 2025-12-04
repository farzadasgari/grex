import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm
from tqdm import tqdm
import warnings
from math import sin, cos, radians, pi

warnings.filterwarnings("ignore")

monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)

lats = monthly_ds.lat.values
lons = monthly_ds.lon.values

years = np.arange(1951, 2026)
n_years = len(years)

months = np.arange(1, 13)
n_months = len(months)

vars_to_save = [
    'prcp', 'tmean', 'tmax', 'tmin', 'pet',
    'spi', 'sti', 'spei',
    'x1', 'sdhi1', 'x2', 'sdhi2'
]
data_dict = {v: np.full((n_years, n_months, len(lats), len(lons)), np.nan) for v in vars_to_save}

for c_idx in tqdm(range(n_cells), desc="Cells"):
    i_lat = lat_idxs[c_idx]
    i_lon = lon_idxs[c_idx]
    lat = lats[i_lat]
    lon = lons[i_lon]

    cell = monthly_ds[['prcp', 'tmean', 'tmax', 'tmin']].sel(lat=lat, lon=lon, method='nearest')

    for m in months:
        m_p = []
        m_tmean = []
        m_tmax = []
        m_tmin = []
        m_pet = []

        for year in years:
            ydata = cell.sel(time=cell['time.year'] == year)
            month_data = ydata.sel(time=ydata['time.month'] == m)
            if month_data.time.size == 0:
                m_p.append(np.nan)
                m_tmean.append(np.nan)
                m_tmax.append(np.nan)
                m_tmin.append(np.nan)
                m_pet.append(np.nan)
                continue

            p_val = month_data['prcp'].values.item()
            tmean_val = month_data['tmean'].values.item()
            tmax_val = month_data['tmax'].values.item()
            tmin_val = month_data['tmin'].values.item()

            lat_rad = radians(lat)
            doy = 31 * (m - 1) + 15
            sol_dec = 0.409 * sin(2 * pi * doy / 365 - 1.39)
            sha = np.arccos(-np.tan(lat_rad) * np.tan(sol_dec))
            ra = 15.39 * (24 / pi) * sha * (0.0835 * sin(2 * pi * doy / 365 - 1.39) + 0.0007)
            trange = tmax_val - tmin_val
            pet_d = 0.0023 * ra * (tmean_val + 17.8) * np.sqrt(max(trange, 0))
            pet_month = pet_d * month_data['time'].dt.daysinmonth.values.item()

            m_p.append(p_val)
            m_tmean.append(tmean_val)
            m_tmax.append(tmax_val)
            m_tmin.append(tmin_val)
            m_pet.append(pet_month)

        m_p = np.array(m_p)
        m_tmean = np.array(m_tmean)
        m_tmax = np.array(m_tmax)
        m_tmin = np.array(m_tmin)
        m_pet = np.array(m_pet)

        valid = ~(np.isnan(m_p) | np.isnan(m_tmean))

        if valid.sum() < 10: continue

        def std_index(arr):
            ranks = rankdata(arr[valid], method='average')
            n = ranks.size
            p = (ranks - 0.44) / (n + 0.12)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return norm.ppf(p)

        spi = std_index(m_p)
        sti = std_index(m_tmean)
        spei = std_index(m_p - m_pet)

        g1_p = (rankdata(m_p[valid], 'average') - 0.44) / (valid.sum() + 0.12)
        g2_t = (rankdata(m_tmean[valid], 'average') - 0.44) / (valid.sum() + 0.12)
        g1_pe = (rankdata((m_p - m_pet)[valid], 'average') - 0.44) / (valid.sum() + 0.12)

        g1_p = np.clip(g1_p, 1e-6, 1 - 1e-6)
        g2_t = np.clip(g2_t, 1e-6, 1 - 1e-6)
        g1_pe = np.clip(g1_pe, 1e-6, 1 - 1e-6)

        x1 = g1_p / g2_t
        x2 = g1_pe / g2_t

        sdhi1 = norm.ppf((rankdata(x1, 'average') - 0.44) / (len(x1) + 0.12))
        sdhi2 = norm.ppf((rankdata(x2, 'average') - 0.44) / (len(x2) + 0.12))

        full_valid_idxs = np.where(valid)[0]
        data_dict['prcp'][full_valid_idxs, m-1, i_lat, i_lon] = m_p[valid]
        data_dict['tmean'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmean[valid]
        data_dict['tmax'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmax[valid]
        data_dict['tmin'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmin[valid]
        data_dict['pet'][full_valid_idxs, m-1, i_lat, i_lon] = m_pet[valid]
        data_dict['spi'][full_valid_idxs, m-1, i_lat, i_lon] = spi
        data_dict['sti'][full_valid_idxs, m-1, i_lat, i_lon] = sti
        data_dict['spei'][full_valid_idxs, m-1, i_lat, i_lon] = spei
        data_dict['x1'][full_valid_idxs, m-1, i_lat, i_lon] = x1
        data_dict['sdhi1'][full_valid_idxs, m-1, i_lat, i_lon] = sdhi1
        data_dict['x2'][full_valid_idxs, m-1, i_lat, i_lon] = x2
        data_dict['sdhi2'][full_valid_idxs, m-1, i_lat, i_lon] = sdhi2

ds_monthly = xr.Dataset(
    {k: (['year', 'month', 'lat', 'lon'], v) for k, v in data_dict.items()},
    coords={'year': years, 'month': months, 'lat': lats, 'lon': lons}
)

ds_monthly.to_netcdf("../dataset/ca_sdhi_monthly_1951_2025.nc")
