# %% [markdown]
# # Test imports

# %% [markdown]
#

# %%
import pandas as pd

# %%
import sys
sys.path.append('../')

# %%
# %load_ext autoreload
# #%autoreload 1
# #%aimport crt1d
# %autoreload 2

# %% [markdown]
# ## General loading

# %%
import crt1d as crt

# %%
crt.config()

# %%
crt.Model

# %%
dir(crt)

# %% [markdown]
# ## Submodules

# %%
crt.leaf_angle

# %%
cd, crs = crt.cases.load_default_case(nlayers=2)
keys = {**cd, **crs}.keys()
keys

# %% [markdown]
# ### Solvers

# %%
crt.solvers

# %%
import inspect
inspect.getfullargspec(crt.solvers.solve_bl)

# %%
inspect.getfullargspec(crt.solvers.solve_4s)

# %%
df_schemes = pd.DataFrame(crt.available_schemes).T

BL_args = set(a for a in df_schemes.loc[df_schemes.ID == 'bl'].args[0])
df_schemes['args_minus_BL'] = df_schemes['args'].apply(lambda x: set(x).symmetric_difference(BL_args))

df_schemes.drop(columns=['args', 'solver'])  # solver memory address not very informative

# %%

# %%
