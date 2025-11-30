import xarray as xr
ds_sdhi = xr.open_dataset("../dataset/ca_sdhi_1951_2025.nc")
ds_scei = xr.open_dataset("../dataset/ca_scei_1951_2025.nc")

print(ds_sdhi)
print(ds_scei)
