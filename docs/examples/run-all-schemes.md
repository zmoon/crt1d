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

* running the model with default case (all schemes)
* plot methods for {class}`~crt1d.Model` class
* {class}`xarray.Dataset` output
* plots for output {class}`xarray.Dataset`s

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt
```

## Examine available schemes

```{code-cell} ipython3
crt.print_config()
```

```{code-cell} ipython3
df_schemes = pd.DataFrame(crt.solvers.AVAILABLE_SCHEMES).T

BL_args = set(a for a in df_schemes.loc[df_schemes.name == "bl"].args[0])
df_schemes["args_minus_BL"] = df_schemes["args"].apply(
    lambda x: list(set(x).symmetric_difference(BL_args))
)

print("B-L args:", ", ".join(BL_args))

df_schemes.drop(columns=["args", "solver"])  # solver memory address not very informative
```

👆 We see that all schemes take at least one more argument than what B–L (`bl`) requires. The B–L scheme makes the black soil assumption and so does not need `soil_r`. Additionally, some schemes have some parameters on top of the required arguments (column `options`).

+++

## Run (just once)
Run the default case (which is loaded automatically when model object is created).

```{code-cell} ipython3
scheme_names = list(crt.solvers.AVAILABLE_SCHEMES)

ms = []
for scheme_name in scheme_names:  # run all available
    m = crt.Model(scheme_name, nlayers=60).run().calc_absorption()
    ms.append(m)

dsets = [m.to_xr() for m in ms]
```

### Examine default case

We plot the leaf area profile, leaf/soil optical property spectra, and top-of-canopy spectral, using the plotting methods attached to the {class}`~crt1d.Model` instance.

```{code-cell} ipython3
m0 = ms[0]

m0.plot_canopy()
m0.plot_leafsoil_spectra()
m0.plot_toc_spectra()
```

## Compare results

In {mod}`crt1d.diagnostics`, there are some functions that can be used to compare a set of model output datasets.

### Plots

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(dsets, band_name="PAR", marker=None)
```

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(
    dsets, band_name="solar", marker=None,
    legend_outside=False, legend_labels="short_name"
)
```

Optionally, we can compare to a reference, in order to better see the differences among schemes.

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(
    dsets, band_name="solar", marker=None,
    legend_outside=False, legend_labels="short_name",
    ref="2s"
)
```

With `ref_relative=True`, the differences are computed as relative error. Otherwise (above), they are absolute. Relative error allows us to better see the differences in the lower regions of the canopy where there is less light.

```{code-cell} ipython3
crt.diagnostics.plot_compare_band(
    dsets, band_name="solar", marker=None,
    legend_outside=False, legend_labels="short_name",
    ref="2s", ref_relative=True
)
```

### Other diagnostic tools

```{code-cell} ipython3
crt.diagnostics.compare_ebal(dsets).astype(float).round(3)
```

## Model output datasets

See [the variables page](/variables.md) for more information about the output variables.

```{code-cell} ipython3
dsets[0]
```

```{code-cell} ipython3
dsets[0]["zm"]
```

```{code-cell} ipython3
# Some schemes generate extra outputs (in addition to the required)
ms[scheme_names.index("zq")].out_extra
```