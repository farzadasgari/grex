import xarray as xr

ds_sdhi = xr.open_dataset("../dataset/ca_sdhi_1951_2025.nc")
ds_scei = xr.open_dataset("../dataset/ca_scei_1951_2025.nc")

ds_sdhi_renamed = ds_sdhi.rename({
    "prcp": "prcp_sdhi",
    "tmean": "tmean_sdhi",
    "tmax": "tmax_sdhi",
    "tmin": "tmin_sdhi",
    "pet": "pet_sdhi",
    "spi": "spi_sdhi",
    "sti": "sti_sdhi",
    "spei": "spei_sdhi",
    "x1": "x1_sdhi",
    "sdhi1": "sdhi1",
    "x2": "x2_sdhi",
    "sdhi2": "sdhi2"
})

ds_scei_renamed = ds_scei.rename({
    "SPI": "spi_scei",
    "STI": "sti_scei",
    "SCEI": "scei"
})

ds_merged = xr.merge([ds_sdhi_renamed, ds_scei_renamed])

ds_merged.attrs["title"] = "California Compound Climate Extremes — SDHI + SCEI (1951–2025)"
ds_merged.attrs["author"] = "Farzad Asgari"
ds_merged.attrs["source"] = "NOAA nClimGrid Gridded Dataset"

out = "../dataset/california.nc"
ds_merged.to_netcdf(out)
