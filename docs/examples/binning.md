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
