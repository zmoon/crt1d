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

# Leaf angles

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt
```

## Leaf inclination angle distributions

```{code-cell} ipython3
named = {
    "spherical": crt.leaf_angle.g_spherical,
    "uniform": crt.leaf_angle.g_uniform,
    "planophile": crt.leaf_angle.g_planophile,
    "erectrophile": crt.leaf_angle.g_erectophile,
    "plagiophile": crt.leaf_angle.g_plagiophile,
}

x = np.linspace(0, np.pi/2, 200)

fig, ax = plt.subplots()

for name, g_fn in named.items():
    ax.plot(np.rad2deg(x), g_fn(x), label=name)

ax.set(
    xlabel=r"Leaf inclination angle $\theta_l$ [deg.]",
    ylabel="Probability density",
)
ax.legend();
```

## $G$
