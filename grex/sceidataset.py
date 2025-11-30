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
