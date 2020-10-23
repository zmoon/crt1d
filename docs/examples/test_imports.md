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


# Test imports

+++

## General loading

```{code-cell}
import crt1d as crt
```

```{code-cell}
crt.print_config()
```

```{code-cell}
crt.Model
```

```{code-cell}
[attr for attr in dir(crt) if not attr.startswith("__")]
```

## Submodules

```{code-cell}
crt.leaf_angle
```

```{code-cell}
crt.data
```

```{code-cell}
p = crt.cases.load_default_case(nlayers=2)
keys = list(p.keys())
keys
```

### Solvers

```{code-cell}
crt.solvers
```

```{code-cell}
import inspect

inspect.getfullargspec(crt.solvers.solve_bl)
```

```{code-cell}
inspect.getfullargspec(crt.solvers.solve_4s)
```

```{code-cell}
import pandas as pd

df_schemes = pd.DataFrame(crt.solvers.AVAILABLE_SCHEMES).T

BL_args = set(a for a in df_schemes.loc[df_schemes.name == "bl"].args[0])
df_schemes["args_minus_BL"] = df_schemes["args"].apply(
    lambda x: set(x).symmetric_difference(BL_args)
)

df_schemes.drop(columns=["args", "solver"])  # solver memory address not very informative
```
