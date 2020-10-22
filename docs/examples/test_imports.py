# %% [markdown]
# # Test imports
# %% [markdown]
# ## General loading
# %%
import crt1d as crt

# %%
crt.print_config()

# %%
crt.Model

# %%
[attr for attr in dir(crt) if not attr.startswith("__")]

# %% [markdown]
# ## Submodules

# %%
crt.leaf_angle

# %%
crt.data

# %%
p = crt.cases.load_default_case(nlayers=2)
keys = list(p.keys())
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
import pandas as pd

df_schemes = pd.DataFrame(crt.solvers.AVAILABLE_SCHEMES).T

BL_args = set(a for a in df_schemes.loc[df_schemes.name == "bl"].args[0])
df_schemes["args_minus_BL"] = df_schemes["args"].apply(
    lambda x: set(x).symmetric_difference(BL_args)
)

df_schemes.drop(columns=["args", "solver"])  # solver memory address not very informative
