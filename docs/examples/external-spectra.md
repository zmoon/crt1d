---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# External spectra sources

`crt1d` comes with some functions that facilitate easy generation of leaf optical property and top-of-canopy irradiance spectra using external programs. These can be found in module {mod}`crt1d.data`.

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt
```

## SPCTRAL2

{func}`crt1d.data.solar_sp2` runs SPCTRAL2 (via [SolarUtils](https://github.com/SunPower/SolarUtils)) and saves the results to an {class}`xarray.Dataset`.

+++

### Example

[Walker Building](https://goo.gl/maps/gXy477ZtfF2cmwB59) in the summer around local solar noon.

```{code-cell} ipython3
lat = 40.793275995962944
lon = -77.86686570952304
dt0 = pd.Timestamp("2021/07/01 13:00")
utcoffset = -4  # EDT

ds = crt.data.solar_sp2(dt0, lat, lon, utcoffset=utcoffset)
ds
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ds.SI_dr.plot(ax=ax, label="direct")
ds.SI_df.plot(ax=ax, label="diffuse")

ax.set(ylabel=f"Spectral irradiance [{ds.SI_dr.units}]")
ax.legend()
```

```{code-cell} ipython3
ds.plot.scatter(x="SI_dr", y="SI_df")
```

### Variations

#### Time of day

```{code-cell} ipython3
dts = pd.date_range("2021/07/01 10:00", "2021/07/01 20:00", freq="15min")

ds = xr.concat(
    [crt.data.solar_sp2(dt, lat, lon, utcoffset=utcoffset) for dt in dts],
    dim="time"
)

def plot(ds, x=None):  
    if x is None:
        x = list(ds.dims)[0]

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    ax1, ax2, ax3, ax4 = axs.flat

    ds.SI_dr.plot(x=x, ax=ax1)
    ds.SI_df.plot(x=x, ax=ax2)

    norm = mpl.colors.LogNorm(vmin=1e-10)
    (ds.SI_dr / ds.SI_dr.isel({x: 0})).plot(x=x, ax=ax3, norm=norm)
    (ds.SI_df / ds.SI_df.isel({x: 0})).plot(x=x, ax=ax4, norm=norm)

    fig.tight_layout()

plot(ds)
```

👆 The second row shows how the spectrum shape changes since the first time shown. It seems that, for the most part, the spectrum shape doesn't change much with time of day on a given day in this model (except around 2.7 μm).

+++

#### Parameters

```{code-cell} ipython3
watvaps = np.linspace(0.1, 10., 100)

ds = xr.concat(
    [crt.data.solar_sp2(dt0, lat, lon, utcoffset=utcoffset, watvap=watvap) for watvap in watvaps],
    dim="watvap"
)

plot(ds)
```

```{code-cell} ipython3
# TODO: others
```
