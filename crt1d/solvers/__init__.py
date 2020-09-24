"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary keyword arguments.

Solver variables
----------------

%(param_table)s

"""

__all__ = ['available_schemes', 'res_keys_all_schemes']


# from collections import namedtuple
# _Vmd = namedtuple("vmd", "")  # TODO: or use dataclass for defaults?

def _load_variable_metadata():
    """Load the variable info from the yml file."""
    from pathlib import Path
    import yaml

    p = Path(__file__).parent/"variables.yml"
    with open(p, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    in_ = []
    for vn, vn_data in data["in"].items():
        vn_data["name"] = vn
        in_.append(vn_data)

    out = []
    for vn, vn_data in data["out"].items():
        vn_data["name"] = vn
        out.append(vn_data)

    return {"in": in_, "out": out}


def _param_entry(d):
    """Form a docstring parameters entry and return as string."""
    # required
    name = d["name"]
    type_ = d["type"]
    desc = d["desc"]
    # optional
    shape = d.get("shape")
    units = d.get("units")
    param = d.get("param", False)

    ind = " "*4  # indent
    s_shape = f"{ind}*shape*: {shape}\n\n" if shape is not None else ""
    s_units = f"{ind}*units*: {units}\n\n" if units is not None else ""
    s_param = ""  # TODO: maybe change?
    s_desc = "\n\n".join(f"{ind}{line}" for line in desc.splitlines())

    return f"""
{name} : {type_}

{s_shape}{s_units}{s_param}{s_desc}
    """


def _list_table_entry(d):
    """Form a reST list table entry and return as string."""
    # required
    name = d["name"]
    type_ = d["type"]
    desc = d["desc"]
    # optional
    shape = d.get("shape")
    units = d.get("units")
    param = d.get("param", False)

    # ind = " "*3  # indent

    s_desc = desc.splitlines()[0]  # temporary hack
    return f"""
   * - ``{name}``
     - {units}
     - ``{shape}``
     - {s_desc}
     """.rstrip()

def _all_params_list_table(vmd_list):
    header = """
.. list-table::
   :widths: 25 25 20 70
   :header-rows: 1

   * - name
     - units
     - shape
     - desc
   """.lstrip()
    return header + "\n".join(_list_table_entry(d) for d in vmd_list)


_vmd = _load_variable_metadata()

# hack module docstring
# include all params
__doc__ %= {
    # "param_in": "\n".join(_param_entry(v) for v in _vmd["in"]), 
    # "param_out": "\n".join(_param_entry(v) for v in _vmd["out"]),
    "param_table": _all_params_list_table(_vmd["in"]),
}


def _get_solver_module_names():
    from pathlib import Path
    solvers_dir = Path(__file__).parent
    return list(p.stem for p in solvers_dir.glob("solve_*.py"))


def _scheme_id_from_module_name(s):
    return s[6:]


_solver_module_names = _get_solver_module_names()
_scheme_IDs = [_scheme_id_from_module_name(mn) for mn in _solver_module_names]

res_keys_all_schemes = ['I_dr', 'I_df_d', 'I_df_u', 'F']
"""
The quantities all schemes must return. Some return others as well.
"""

available_schemes = { scheme_ID: {} for scheme_ID in _scheme_IDs }
"""
Dictionary of available canopy RT schemes, where keys are the scheme ID,
and values are ``short_name``, ``long_name``, ``solver`` (the associated solver function).
"""

def _construct_scheme_dicts():
    """Extract info from the individual solver modules to fill `available_schemes` with info."""
    from importlib import import_module
    import inspect
    import warnings

    # extract info
    for name, (scheme_ID, scheme_dict) in zip(_solver_module_names, available_schemes.items()):

        scheme_dict['module_name'] = name
        scheme_dict['ID'] = scheme_ID

        # load module and import things 
        module = import_module(f'.{name}', package='crt1d.solvers')
        solve_fun_name = f'solve_{scheme_ID}'  # or could change the fn names to just be 'solve' and import alias here
        solver = getattr(module, solve_fun_name)  # get solver function
        short_name = getattr(module, 'short_name', scheme_ID)
        long_name = getattr(module, 'long_name', '')
        if not long_name:
            warnings.warn(f"`long_name` not defined for solver module {name!r}")

        # set
        scheme_dict['short_name'] = short_name
        scheme_dict['long_name'] = long_name
        scheme_dict['solver'] = solver

    # extract signature
    for scheme_dict in available_schemes.values():
        fullargspec = inspect.getfullargspec(scheme_dict['solver'])
        # scheme_dict['args'] = fullargspec.args
        scheme_dict['args'] = fullargspec.kwonlyargs
        # scheme_dict['args'] = [k for k in fullargspec.kwonlyargs if k not in fullargspec.kwonlydefaults]
        kwd = fullargspec.kwonlydefaults
        if kwd is not None:
            for k in kwd:
                scheme_dict['args'].remove(k)

_construct_scheme_dicts()


