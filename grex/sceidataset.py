import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm, multivariate_normal
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def gringorten_cdf(arr):
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return out
    ranks = rankdata(arr[valid], method="average")
    n = valid.sum()
    g = (ranks - 0.44) / (n + 0.12)
    g = np.clip(g, 1e-6, 1 - 1e-6)
    out[valid] = g
    return out


def standardized_index_from_series(arr):
    g = gringorten_cdf(arr)
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(g)
    out[valid] = norm.ppf(g[valid])
    return out


def joint_prob_dry_hot(spi_vals, sti_vals):
    spi = np.asarray(spi_vals, dtype=float)
    sti = np.asarray(sti_vals, dtype=float)
    p = np.full_like(spi, np.nan)
    valid = ~(np.isnan(spi) | np.isnan(sti))
    if valid.sum() < 3: return p
    u1 = spi[valid]
    u2 = sti[valid]
    rho = np.corrcoef(u1, u2)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)
    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)
    p_valid = np.zeros_like(u1)
    for i in range(len(u1)):
        u1_i = u1[i]
        u2_i = u2[i]
        p_u1 = norm.cdf(u1_i)
        p_u1_u2 = mvn.cdf([u1_i, u2_i])
        p_valid[i] = p_u1 - p_u1_u2
    p_valid = np.clip(p_valid, 1e-8, 1 - 1e-8)
    p[valid] = p_valid
    return p
