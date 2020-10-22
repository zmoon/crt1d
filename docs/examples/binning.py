# %% [markdown]
# # Demonstrate binning functionality
#
# In module `crt1d.spectra`.
# %%
import numpy as np

import crt1d as crt

# %% [markdown]
# ## Load included spectra

# %%
ds0 = crt.data.load_default_sp2()
ds0_l = crt.data.load_ideal_leaf().dropna(dim="wl")

# %% [markdown]
# ## Show different binnings

# %%
bins1 = np.linspace(0.3, 2.5, 20)
ds1 = crt.spectra.smear_si(
    ds0, bins1
)  # special function needed to calculate smeared I by smearing SI
crt.spectra.plot_binned_ds(ds0, ds1, yname="SI_df")

# %%
ds1_l = crt.spectra.smear_ds(ds0_l, bins1)
crt.spectra.plot_binned_ds(ds0_l, ds1_l, yname="rl")

# %%
bins2 = np.linspace(0.3, 2.6, 50)
ds2_l = crt.spectra.smear_ds(ds0_l, bins2)
crt.spectra.plot_binned_ds(ds0_l, ds2_l, yname="rl")

# %% [markdown]
# Note that the bins do not have to be constant-width--variable grid will also work!
