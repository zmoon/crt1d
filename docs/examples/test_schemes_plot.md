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

# Run each scheme

We test the following:

* running the model with default case
* plot methods for Model class
* output xr
* plots for Model class
* plots for output xr

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt

#%matplotlib notebook
%matplotlib inline
```

## Run (just once)
Run the default case (which is loaded automatically when model object is created).

```{code-cell}
ms = []
for scheme_ID in crt.solvers.AVAILABLE_SCHEMES.keys():  # run all available
    m = crt.Model(scheme_ID, nlayers=60)
    m.run()
    m.calc_absorption()
    ms.append(m)

dsets = [m.to_xr() for m in ms]
```

### Examine default case

Leaf area profile and leaf angle dist, using the plotting methods attached to the `Model` instance.

```{code-cell}
ms[0].plot_canopy()
```

```{code-cell}
ms[0].plot_leafsoil_spectra()
```

```{code-cell}
ms[0].plot_toc_spectra()
```

## Plots of results

```{code-cell}
crt.diagnostics.plot_compare_band(dsets, band_name="PAR")
```

```{code-cell}
crt.diagnostics.plot_compare_band(dsets, band_name="solar")
```

### Other diagnostic tools

```{code-cell}
crt.diagnostics.df_E_balance(dsets).astype(float).round(3)
```

## Datasets and their variables

```{code-cell}
dsets[0]
```

```{code-cell}
dsets[0]["zm"]
```

```{code-cell}
m.out_extra.keys()
```
