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

import crt1d as crt
```

## Load included spectra

```{code-cell} ipython3
ds0 = crt.data.load_default_sp2()
ds0_l = crt.data.load_ideal_leaf().dropna(dim="wl")
```

## Different binnings

Although the main purpose of the "smear" functions is to provide the ability to go to lower spectral resolution while conserving certain properties, they also work as interpolators when a higher resolution grid is used.

```{code-cell} ipython3
bins1 = np.linspace(0.3, 2.5, 20)
ds1 = crt.spectra.smear_si(ds0, bins1)
# ^ we use a special function to calculate smeared I by smearing SI
crt.spectra.plot_binned_ds(ds0, ds1, yname="SI_df")
```

```{code-cell} ipython3
ds1_l = crt.spectra.smear_ds(ds0_l, bins1)
crt.spectra.plot_binned_ds(ds0_l, ds1_l, yname="rl")
```

```{code-cell} ipython3
bins2 = np.linspace(0.3, 2.6, 200)
ds2_l = crt.spectra.smear_ds(ds0_l, bins2)
crt.spectra.plot_binned_ds(ds0_l, ds2_l, yname="rl")
```

```{code-cell} ipython3
bins3 = np.linspace(0.3, 2.6, 4)
ds3_l = crt.spectra.smear_ds(ds0_l, bins3, method="tuv")
crt.spectra.plot_binned_ds(ds0_l, ds3_l, yname="rl")
```

```{code-cell} ipython3
ds3_l_2 = crt.spectra.smear_ds(ds0_l, bins3, method="avg_optical_prop")
crt.spectra.plot_binned_ds(ds0_l, ds3_l_2, yname="rl")
```

```{note}
The bins do not have to be constant-width---variable grid will also work!
```

## Planck function

[Planck](https://en.wikipedia.org/wiki/Planck%27s_law) radiance can be used to weight values
when smearing the spectra. This is important for albedo, for example, because albedo in some waveband (such as visible or shortwave) depends on both the albedo spectrum and the spectrum of the illuminating light.

```{code-cell} ipython3
wl_um = np.linspace(0.3, 2.6, 200)
fig, ax = plt.subplots()
for T_K in [4000, 5000, 5800, 6000]:
    ax.plot(wl_um, crt.spectra.l_wl_planck(T_K, wl_um), label=T_K)
ax.set(
    xlabel=r"Wavelength $\lambda$ [μm]",
    ylabel=r"Spectral radiance $L(\lambda)$ [W sr$^{-1}$ m$^{-2}$ m$^{-1}$]",
)
ax.legend(title="Blackbody temperature (K)")
```

## Optical properties

Smearing optical properties as shown above is a bit unfair(?). As mentioned in the previous section, reflectance/transmittance are like albedo in that they depend on the incoming light as well. We can account for this by weighting with a light spectrum when averaging the optical property spectra.

Below, we can see that when we compute the average reflectance, over certain regions this doesn't matter much (PAR/visible). In the NIR, however, there is a strong preference for smaller wavelengths, which overall weights leaf reflectance towards higher values.

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
data = (
    ds1.sel(wl=ds1_l.wl.values, method="nearest").I_dr +
    ds1.sel(wl=ds1_l.wl.values, method="nearest").I_df
)
lights = ["planck", "uniform", data]

for bounds in boundss:
    print(bounds)
    for light in lights:
        ybar = crt.spectra.avg_optical_prop(
            ds1_l.rl, bounds, xe=ds1_l.wle, light=light
        )
        if isinstance(light, str):
            print(f" {light}: {ybar:.4g}")
        else:
            print(f" data: {ybar:.4g}")
```

👆 Notice that over the whole spectrum, the values for 'uniform' and 'planck' are very similar to the calculations on the pre-smeared spectrum.
