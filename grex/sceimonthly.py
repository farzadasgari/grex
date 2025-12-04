import xarray as xr
import numpy as np
from scipy.stats import norm, rankdata, multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def standardize(series):
    valid = ~np.isnan(series)
    if valid.sum() < 10: return np.full_like(series, np.nan)
    ranks = rankdata(series[valid], method='average')
    n = valid.sum()
    p = (ranks - 0.44) / (n + 0.12)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    out = np.full_like(series, np.nan)
    out[valid] = norm.ppf(p)
    return out

def compute_scei_for_month(prcp_series, temp_series):
    spi = standardize(prcp_series)
    sti = standardize(temp_series)
    valid = ~(np.isnan(spi) | np.isnan(sti))
    if valid.sum() < 10: return np.full_like(prcp_series, np.nan)
    
    u1 = spi[valid]
    u2 = sti[valid]
    
    rho = np.corrcoef(u1, u2)[0,1]
    if not np.isfinite(rho): rho = 0.0
    rho = np.clip(rho, -0.999, 0.999)
    cov = [[1, rho], [rho, 1]]
    try:
        mvn = multivariate_normal([0, 0], cov, allow_singular=True)
        joint_cdf = np.array([mvn.cdf([a, b]) for a, b in zip(u1, u2)])
    except: joint_cdf = norm.cdf(u1) * norm.cdf(u2)
    
    p_dry_and_hot = norm.cdf(u1) - joint_cdf
    p_dry_and_hot = np.clip(p_dry_and_hot, 1e-8, 1 - 1e-8)
    ranks = rankdata(p_dry_and_hot, method='average')
    n = len(p_dry_and_hot)
    p_final = (ranks - 0.44) / (n + 0.12)
    scei = norm.ppf(p_final)
    
    result = np.full_like(prcp_series, np.nan)
    result[valid] = scei
    return result

def process_cell(i_lat, i_lon):
    cell = monthly_ds.sel(lat=lats[i_lat], lon=lons[i_lon], method='nearest')[['prcp', 'tmean']]
    result = np.full((75, 12, 5), np.nan)    
    for month in range(1, 13):
        month_data = cell.where(cell.time.dt.month == month, drop=True)
        if len(month_data.time) == 0: continue
            
        prcp = np.array([x.item() if hasattr(x, 'item') else x for x in month_data.prcp.values])
        temp = np.array([x.item() if hasattr(x, 'item') else x for x in month_data.tmean.values])
        
        years_in_data = month_data.time.dt.year.values
        year_indices = years_in_data - 1951
        
        if len(prcp) < 10 or np.isnan(prcp).all(): continue
        scei_values = compute_scei_for_month(prcp, temp)
        
        valid_years = year_indices[~np.isnan(scei_values)]
        valid_scei = scei_values[~np.isnan(scei_values)]
        
        if len(valid_years) == 0: continue
        result[valid_years, month-1, 0] = prcp[~np.isnan(scei_values)]
        result[valid_years, month-1, 1] = temp[~np.isnan(scei_values)]
        result[valid_years, month-1, 2] = standardize(prcp)[~np.isnan(scei_values)]
        result[valid_years, month-1, 3] = standardize(temp)[~np.isnan(scei_values)]
        result[valid_years, month-1, 4] = valid_scei
    
    return i_lat, i_lon, result

if __name__ == '__main__':
    monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc").load()
    warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")
    
    mask = ~np.isnan(warm_da).any(dim='window')
    lat_idx, lon_idx = np.where(mask)
    lats = monthly_ds.lat.values
    lons = monthly_ds.lon.values
        
    results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(process_cell)(i, j) for i, j in tqdm(zip(lat_idx, lon_idx), total=len(lat_idx), desc="SCEI")
    )
    
    ny, nm, nlats, nlons = 75, 12, len(lats), len(lons)
    final = {var: np.full((ny, nm, nlats, nlons), np.nan) for var in ['prcp','tmean','spi','sti','scei']}
    
    for i_lat, i_lon, arr in results:
        for v in range(5):
            varname = ['prcp','tmean','spi','sti','scei'][v]
            final[varname][:,:,i_lat,i_lon] = arr[:,:,v]
    
    ds = xr.Dataset(
        {k: (['year','month','lat','lon'], v) for k, v in final.items()},
        coords={
            'year': range(1951, 2026),
            'month': range(1, 13),
            'lat': lats,
            'lon': lons
        }
    )

    output_file = "../dataset/ca_scei_monthly_1951_2025.nc"
    ds.to_netcdf(output_file, encoding={v: {"zlib": True, "complevel": 5} for v in ds})
