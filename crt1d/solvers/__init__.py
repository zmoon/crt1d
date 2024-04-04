"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary keyword arguments.

See docs/solvers for descriptions of scheme input/output variables.
"""

# note: schemes don't require any code from this module to run standalone
#       some require module common, all need K_b, psi, lai, leaf_r, leaf_t
#
# note that __all__ is currently modified below when loading the solvers
__all__ = ["AVAILABLE_SCHEMES", "RET_KEYS_ALL_SCHEMES"]

from ..variables import VMD as _vmd


def _get_solver_module_names():
    from pathlib import Path

    solvers_dir = Path(__file__).parent
    return sorted(p.stem for p in solvers_dir.glob("_solve_*.py"))


def _scheme_id_from_module_name(s):
    return s[7:]


_solver_module_names = _get_solver_module_names()
_scheme_names = [_scheme_id_from_module_name(mn) for mn in _solver_module_names]
_solver_modules = dict(zip(_scheme_names, _solver_module_names))

CANOPY_RAD_STATE_INPUT_KEYS = list(_vmd.intent("in"))  # keys only

RET_KEYS_ALL_SCHEMES = ["I_dr", "I_df_d", "I_df_u", "F"]
"""
The quantities all schemes must return. Some return others as well.
"""

# Make sure the required return keys are in vmd
assert all(k in _vmd.intent("out") for k in RET_KEYS_ALL_SCHEMES)


AVAILABLE_SCHEMES = {scheme_name: {} for scheme_name in _scheme_names}
"""
Dictionary of available canopy RT schemes, where keys are the scheme name/ID,
and values are dicts of scheme info: ``short_name``, ``long_name``,
``solver`` (the associated solver function), etc.
"""

# TODO: fn to load one scheme at a time, to support user adding their own from outside the package


def _construct_scheme_dicts():
    """Extract info from the individual solver modules to fill `AVAILABLE_SCHEMES` with info."""
    import inspect
    import warnings
    from importlib import import_module

    # extract info
    for name, scheme_dict in AVAILABLE_SCHEMES.items():
        module_name = _solver_modules[name]

        scheme_dict["module_name"] = module_name
        scheme_dict["name"] = name  # `name` is the primary identifier

        # load module and import things
        module = import_module(f".{module_name}", package="crt1d.solvers")
        solve_fun_name = f"solve_{name}"
        # ^ or could change the fn names to just be 'solve' and import alias here
        solver = getattr(module, solve_fun_name)  # get solver function
        short_name = getattr(module, "short_name", name)
        long_name = getattr(module, "long_name", "")
        if not long_name:
            warnings.warn(f"`long_name` not defined for solver module {module_name!r}")

        # set
        scheme_dict["short_name"] = short_name
        scheme_dict["long_name"] = long_name
        scheme_dict["solver"] = solver

    # extract signature
    drop_list = []
    for name, scheme_dict in AVAILABLE_SCHEMES.items():
        fullargspec = inspect.getfullargspec(scheme_dict["solver"])
        # scheme_dict['args'] = fullargspec.args
        scheme_dict["args"] = fullargspec.kwonlyargs
        scheme_dict["options"] = []
        # scheme_dict['args'] = [k for k in fullargspec.kwonlyargs if k not in fullargspec.kwonlydefaults]
        kwd = fullargspec.kwonlydefaults
        if kwd is not None:
            for k in kwd:
                scheme_dict["args"].remove(k)
                scheme_dict["options"].append(k)
        # drop scheme and warn if args don't match with expected
        if any(k not in CANOPY_RAD_STATE_INPUT_KEYS for k in scheme_dict["args"]):
            invalid_kwargs = [
                k for k in scheme_dict["args"] if k not in CANOPY_RAD_STATE_INPUT_KEYS
            ]
            warnings.warn(
                f"Some arguments for scheme {name!r} not compatible with the expected:\n"
                f"  {', '.join(CANOPY_RAD_STATE_INPUT_KEYS)}\n"
                f"As a result, {name!r} will not be loaded.\n"
                "Invalid keys:\n"
                f"  {', '.join(invalid_kwargs)}"
            )
            drop_list.append(name)
    # drop?
    for name in drop_list:
        AVAILABLE_SCHEMES.pop(name)


_construct_scheme_dicts()

# add solver functions to the solvers module namespace
for _solver_dict in AVAILABLE_SCHEMES.values():  # get this out of module-level scope
    solver = _solver_dict["solver"]
    solver_name = solver.__name__
    globals().update({solver_name: solver})
    # also to __all__
    __all__.append(solver_name)
    # ^ static analyzers don't seem to account for this
