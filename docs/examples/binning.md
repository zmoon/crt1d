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

# Binning spectra

Module {mod}`crt1d.spectra` provides some tools for working with spectra.
By binning or smearing spectra, we can use spectra together that were originally on different grids.
Also, we can decrease canopy RT computational time by reducing spectral resolution.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import crt1d as crt
```

## Load included spectra

```{code-cell} ipython3
ds0 = crt.data.load_default_sp2()
ds0_l = crt.data.load_ideal_leaf().dropna(dim="wl")
```

## Different binnings

### Solar

Although the main purpose of the "smear" functions is to provide the ability to go to lower spectral resolution while conserving certain properties, they also work as interpolators when a higher resolution grid is used.

```{note}
The bins do not have to be equal-width---variable grid will also work!
```

```{code-cell} ipython3
ds0
```

```{code-cell} ipython3
binss = [
    [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.5],  # variable dx is supported
    np.linspace(0.3, 2.5, 8),
    np.linspace(0.3, 2.5, 21),
    np.linspace(0.3, 2.5, 201),
]

fig, ax = plt.subplots(2, 2, figsize=(11, 7))

for bins, ax in zip(binss, ax.flat):
    # The spectral variables have coord `wl0`
    crt.spectra.plot_binned_ds(ds0, crt.spectra.smear_ds(ds0, bins, xname="wl0"), yname="SI_df", xtight="bins", ax=ax)
```

```{warning}
Irradiance (W m-2) should not be smeared directly (as we see below).
Instead we use a special routine to calculate in-bin irradiance by smearing *spectral* irradiance (W m-2 μm-1).
```

+++

If we smear irradiance directly, the total irradiance is essentially proportional to the number of bins used. For equal-width bins, it turns out to be exactly proportional.

```{code-cell} ipython3
tot0 = ds0.I_dr.sum().values
tot = []
nbands = np.arange(1, 200, 10)
for n in nbands:
    tot.append(crt.spectra.smear_ds(ds0, np.linspace(0.3, 4.0, n)).I_dr.sum().values)

res = stats.linregress(nbands, tot)

fig, ax = plt.subplots()
ax.plot(nbands, tot, ".-")
ax.text(0.03, 0.96, f"Original total: {tot0:.4g}\nr$^2 = {res.rvalue**2:.3f}$", ha="left", va="top", transform=ax.transAxes)
ax.set(xlabel="# of bands", ylabel="Total irradiance [W m-2]");
```

Instead, for computing irradiance in bins, {func}`crt1d.spectra.smear_si` can be used for convenience. It smears *spectral* irradiance and then multiplies by bin width to give irradiance.

+++

### Leaf/soil optical properties

Smearing leaf/soil optical properties directly comes with its own issues.

Reflectance/transmittance are like albedo in that they depend on the incoming light as well. We can account for this by weighting with a light spectrum when averaging the optical property spectra.

Below, we can see that when we compute the average reflectance, over certain regions this doesn't matter much (PAR/visible). In the NIR, however, there is a strong preference for smaller wavelengths, which overall weights leaf reflectance towards higher values.

```{code-cell} ipython3
ds0_l
```

```{code-cell} ipython3
bins2 = np.linspace(0.3, 2.6, 21)
ds2_l = crt.spectra.smear_ds(ds0_l, bins2)
crt.spectra.plot_binned_ds(ds0_l, ds2_l, yname="rl")
```

The influence of accounting for the incoming light spectrum is most apparent when the bins are large and few.

```{code-cell} ipython3
bins3 = np.linspace(0.3, 2.6, 4)

fig, axs = plt.subplots(2, 1, figsize=(6, 6))

for method, ax in zip(["tuv", "avg_optical_prop"], axs.flat):
    crt.spectra.plot_binned_ds(ds0_l, crt.spectra.smear_ds(ds0_l, bins3, method=method), yname="rl", ax=ax)
```

## Average optical properties

```{code-cell} ipython3
# On the original leaf reflectance spectrum
boundss = [(0.4, 0.7), (0.7, 1.0), (1.0, 2.5), (0.3, 2.5)]
lights = ["planck", "uniform"]

for bounds in boundss:
    print(bounds)
    for light in lights:
        ybar = crt.spectra.avg_optical_prop(
            ds0_l.rl, bounds, x=ds0_l.wl, light=light
        )
        print(f" {light}: {ybar:.4g}")
```

```{code-cell} ipython3
# On the smeared/binned leaf reflectance spectrum
ds2 = crt.spectra.smear_ds(ds0, bins2)
data = (
    ds2.sel(wl=ds2_l.wl.values, method="nearest").I_dr +
    ds2.sel(wl=ds2_l.wl.values, method="nearest").I_df
)
lights = ["planck", "uniform", data]

for bounds in boundss:
    print(bounds)
    for light in lights:
        ybar = crt.spectra.avg_optical_prop(
            ds2_l.rl, bounds, xe=ds2_l.wle, light=light
        )
        if isinstance(light, str):
            print(f" {light}: {ybar:.4g}")
        else:
            print(f" data: {ybar:.4g}")
```

👆 Notice that over the whole spectrum, the values for 'uniform' and 'planck' are very similar to the calculations on the pre-smeared spectrum.

+++

## Planck function

[Planck](https://en.wikipedia.org/wiki/Planck%27s_law) radiance is one of the `light` options can be used to weight values
when smearing the spectra with the `avg_optical_prop` option. This is important for albedo, for example, because albedo in some waveband (such as visible or shortwave) depends on both the albedo spectrum and the spectrum of the illuminating light.

```{code-cell} ipython3
wl_um = np.linspace(0.3, 2.6, 200)
fig, ax = plt.subplots()
for T_K in [4000, 5000, 5800, 6000]:
    ax.plot(wl_um, crt.spectra.l_wl_planck(T_K, wl_um), label=T_K)
ax.set(
    xlabel=r"Wavelength $\lambda$ [μm]",
    ylabel=r"Spectral radiance $L(\lambda)$ [W sr$^{-1}$ m$^{-2}$ m$^{-1}$]",
)
ax.legend(title="Blackbody temperature (K)");
```
