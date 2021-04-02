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

# Run each scheme

We test the following:

* running the model with default case
* plot methods for {class}`~crt1d.Model` class
* :class:`xarray.Dataset` output
* plots for output `xarray.Dataset`s

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt
```

## Run (just once)
Run the default case (which is loaded automatically when model object is created).

```{code-cell} ipython3
scheme_names = list(crt.solvers.AVAILABLE_SCHEMES)

ms = []
for scheme_name in scheme_names:  # run all available
    m = crt.Model(scheme_name, nlayers=60)
    m.run()
    m.calc_absorption()
    ms.append(m)

dsets = [m.to_xr() for m in ms]
```

### Examine default case

Leaf area profile and leaf angle dist, using the plotting methods attached to the {class}`~crt1d.Model` instance.

```{code-cell} ipython3
ms[0].plot_canopy()
```

```{code-cell} ipython3
ms[0].plot_leafsoil_spectra()
```

```{code-cell} ipython3
ms[0].plot_toc_spectra()
```

## Plots of results

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(dsets, band_name="PAR")
```

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(dsets, band_name="solar")
```

### Other diagnostic tools

```{code-cell} ipython3
crt.diagnostics.compare_ebal(dsets).astype(float).round(3)
```

## Datasets and their variables

```{code-cell} ipython3
dsets[0]
```

```{code-cell} ipython3
dsets[0]["zm"]
```

```{code-cell} ipython3
# some have extra outputs
ms[scheme_names.index("zq")].out_extra
```
