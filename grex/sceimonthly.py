import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm, multivariate_normal
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def gringorten_cdf(arr):
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return np.full_like(arr, np.nan)
    ranks = rankdata(arr[valid], method="average")
    n = valid.sum()
    p = (ranks - 0.44) / (n + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = np.full_like(arr, np.nan)
    out[valid] = p
    return out

def standardized_index(arr):
    p = gringorten_cdf(arr)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return out
    out[valid] = norm.ppf(p[valid])
    return out

def joint_prob_dry_and_hot(spi_vals, sti_vals):
    spi = np.asarray(spi_vals, dtype=float)
    sti = np.asarray(sti_vals, dtype=float)
    valid = ~(np.isnan(spi) | np.isnan(sti))
    if valid.sum() < 10: return np.full_like(spi, np.nan)
    
    u1 = spi[valid]
    u2 = sti[valid]

    rho = np.corrcoef(u1, u2)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)
    cov = [[1, rho], [rho, 1]]
    mvn = multivariate_normal(mean=[0, 0], cov=cov)

    p_joint_cdf = np.array([mvn.cdf([u1[i], u2[i]]) for i in range(len(u1))])
    p_dry = norm.cdf(u1)
    p_dry_and_hot = p_dry - p_joint_cdf

    p_dry_and_hot = np.clip(p_dry_and_hot, 1e-8, 1 - 1e-8)

    result = np.full_like(spi, np.nan)
    result[valid] = p_dry_and_hot
    return result

def scei_from_prob(p):
    g = gringorten_cdf(p)
    out = np.full_like(g, np.nan)
    valid = ~np.isnan(g)
    if valid.sum() < 3: return out
    out[valid] = norm.ppf(g[valid])
    return out

monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)

lats = monthly_ds.lat.values
lons = monthly_ds.lon.values
years = np.arange(1951, 2026)
months = np.arange(1, 13)

vars_to_save = ['prcp', 'tmean', 'spi', 'sti', 'scei']
data_dict = {
    v: np.full((len(years), len(months), len(lats), len(lons)), np.nan)
    for v in vars_to_save
}

for c_idx in tqdm(range(n_cells), desc="Grid Cells", unit="cell"):
    i_lat = lat_idxs[c_idx]
    i_lon = lon_idxs[c_idx]

    cell = monthly_ds[['prcp', 'tmean']].sel(lat=lats[i_lat], lon=lons[i_lon], method='nearest')

    for m in months:
        values_prcp = []
        values_temp = []

        for y in years:
            sel = cell.sel(time=(cell['time.year'] == y) & (cell['time.month'] == m))
            if sel.sizes['time'] == 0:
                values_prcp.append(np.nan)
                values_temp.append(np.nan)
                continue
            values_prcp.append(float(sel.prcp.values))
            values_temp.append(float(sel.tmean.values))

        prcp_arr = np.array(values_prcp)
        temp_arr = np.array(values_temp)
        valid_mask = ~(np.isnan(prcp_arr) | np.isnan(temp_arr))

        if valid_mask.sum() < 10: continue
        spi_full = standardized_index(prcp_arr)
        sti_full = standardized_index(temp_arr)

        p_dry_hot = joint_prob_dry_and_hot(spi_full, sti_full)
        scei_full = scei_from_prob(p_dry_hot)

        valid_years_idx = np.where(valid_mask)[0]
        
        data_dict['prcp'][valid_years_idx, m-1, i_lat, i_lon] = prcp_arr[valid_mask]
        data_dict['tmean'][valid_years_idx, m-1, i_lat, i_lon] = temp_arr[valid_mask]
        data_dict['spi'][valid_years_idx, m-1, i_lat, i_lon] = spi_full[valid_mask]
        data_dict['sti'][valid_years_idx, m-1, i_lat, i_lon] = sti_full[valid_mask]
        data_dict['scei'][valid_years_idx, m-1, i_lat, i_lon] = scei_full[valid_mask]

ds_scei_monthly = xr.Dataset(
    {k: (['year', 'month', 'lat', 'lon'], v) for k, v in data_dict.items()},
    coords={'year': years, 'month': months, 'lat': lats, 'lon': lons}
)


ds_scei_monthly.to_netcdf("../dataset/ca_scei_monthly_1951_2025.nc")
