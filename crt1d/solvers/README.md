
## Adding new schemes

Schemes should be added as *modules* with names of the form: `solve_{scheme_id}.py` (`scheme_id` will be extracted from the module name).

Schemes should define the following *module-level variables*:

* `long_name` -- a more descriptive name, but short enough to be used in plot labels, e.g. `'Dickinson–Sellers two-stream'`
* `short_name` -- a short version, only needed if different from `scheme_id`; e.g. `'2s'`

:point_up: the names should be unique; 

Scheme *modules* must contain a *function* with the same name as the module that:

* takes keyword arguments (and only keyword arguments! so that order doesn't matter when passing args in; see which names can be used in the `solvers` module docstring)
* returns a dict, including, but not limited to, the four required outputs (direct irradiance, downward/upward diffuse irradiance, actinic flux); see `solvers` module docstring for names)
* has a docstring that includes some info about the scheme, key references, etc. (module docstring not necessary)
