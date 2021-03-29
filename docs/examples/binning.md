---
kernelspec:
  display_name: Python 3
  name: python3
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
---

# Demonstrate binning functionality

In module {mod}`crt1d.spectra`.

```{code-cell}
import numpy as np

import crt1d as crt
```

## Load included spectra

```{code-cell}
ds0 = crt.data.load_default_sp2()
ds0_l = crt.data.load_ideal_leaf().dropna(dim="wl")
```

## Show different binnings

```{code-cell}
bins1 = np.linspace(0.3, 2.5, 20)
ds1 = crt.spectra.smear_si(ds0, bins1)
# ^ we use a special function to calculate smeared I by smearing SI
crt.spectra.plot_binned_ds(ds0, ds1, yname="SI_df")
```

```{code-cell}
ds1_l = crt.spectra.smear_ds(ds0_l, bins1)
crt.spectra.plot_binned_ds(ds0_l, ds1_l, yname="rl")
```

```{code-cell}
bins2 = np.linspace(0.3, 2.6, 50)
ds2_l = crt.spectra.smear_ds(ds0_l, bins2)
crt.spectra.plot_binned_ds(ds0_l, ds2_l, yname="rl")
```

Note that the bins do not have to be constant-width---variable grid will also work!
