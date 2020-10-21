# %% [markdown]
# # Run each scheme
#
# We test the following:
#
# * running the model with default case
# * plot methods for Model class
# * output xr
# * plots for Model class
# * plots for output xr
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import crt1d as crt

# #%matplotlib notebook
# %matplotlib inline

# %% [markdown]
# ## Run (just once)
# Run the default case (which is loaded automatically when model object is created).

# %%
schemes_to_test = ["bl", "2s", "4s", "zq"]

ms = []
for scheme_ID in schemes_to_test:
    m = crt.Model(scheme_ID, nlayers=60)
    # m.plot_canopy()
    m.run()
    m.calc_absorption()
    ms.append(m)

dsets = [m.to_xr() for m in ms]

# %% [markdown]
# ### Examine default case
#
# Leaf area profile and leaf angle dist, using the plotting methods attached to the `Model` instance.

# %%
ms[0].plot_canopy()

# %%
ms[0].plot_leafsoil_spectra()

# %%
ms[0].plot_toc_spectra()

# %% [markdown]
# ## Plots of results

# %%
crt.diagnostics.plot_compare_band(dsets, band_name="PAR")

# %%
crt.diagnostics.plot_compare_band(dsets, band_name="solar")

# %% [markdown]
# ### Other diagnostic tools

# %%
crt.diagnostics.df_E_balance(dsets).astype(float).round(3)

# %% [markdown]
# ## Datasets and their variables

# %%
dsets[0]

# %%
dsets[0]["zm"]

# %%
m.out_extra.keys()
