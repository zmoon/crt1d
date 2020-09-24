# %% [markdown]
# # Run each scheme
#
# Using this Notebook to test the module, so using IPy extension `autoreload`. 
#
# Test the following:
#
# * running 
# * plot methods for Model class
# * output xr
# * plots for Model class
# * plots for output xr

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# %%
# #%matplotlib notebook
# %matplotlib inline

# %%
import sys
sys.path.append('../')
# %load_ext autoreload
# %autoreload 2
import crt1d as crt

# %% [markdown]
# ## Run (just once)
# Run the default case (which is loaded automatically when model object is created).

# %%
schemes_to_test = ['bl', 'bf', 'gd', '2s', 'zq', '4s']
#schemes_to_test = ['2s']
#schemes_to_test = ['bl', '2s', '4s', 'zq']
#schemes_to_test = ['bl', '2s', '4s', 'bf']
#schemes_to_test = ['2s', '4s']

ms = []
for scheme_ID in schemes_to_test:
    m = crt.Model(scheme_ID, nlayers=60)
    #m.plot_canopy()
    m.run()
    ms.append(m)
    
dsets = [m.to_xr() for m in ms]

# %% [markdown]
# ### Examine default case
#
# Leaf area profile and leaf angle dist

# %%
ms[0].plot_canopy()

# %% [markdown]
# ## Plots of results

# %%
crt.model.plot_PAR(dsets)

# %%
crt.model.plot_solar(dsets)

# %% [markdown]
# ### Other diagnostic tools

# %%
df = crt.model.create_E_closure_table(dsets)
df.astype(float).round(2)

# %%
m.crs.keys()

# %% [markdown]
# ## Datasets and their variables

# %%
dsets[0]

# %%
dsets[0]['zm']

# %%
m.extra_solver_results.keys()
