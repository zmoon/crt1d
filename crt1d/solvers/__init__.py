"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary arguments using the cnpy_rad_state and cnpy_descrip 
dictionary inputs.

For now leaving all as looping over all available wavelengths.


Required inputs
---------------
psi : float
    Solar zenith angle (SZA; radians)
I_dr0_all, Idf0_all : np array (n_wl) 
    direct and diffuse irradiances (W/m^2) at top of canopy
    at all wavelengths
    not spectral! (which would be W/m^2/um)
wl, dwl : np array (n_wl)
    wavelength band centers and bandwidths (m)
    for the top-of-canopy irradiances and other spectral quantities
    no longer needed
lai : np array (n_z)
    LAI profile
    lai[0] = total LAI
    lai[-1] = 0 (canopy-atmos interface layer)
leaf_t, leaf_r : np array (n_wl)
    leaf element spectral transmissivity and reflectivity (unitless)
green : float
    canopy green-ness factor (0, 1], to weight the leaf_t,_r
K_b_fn : K_b_fn(psi) 
    the function to be used to compute K_b
    (depends on which leaf angle distribution is used
        so must be passed)


G_fn : G_fn(psi)
    leaf angle dist factor
    G = K_b * mu 
mean_leaf_angle : float
    mean leaf inclination angle (deg.)



Returns
-------
I_dr_all, I_df_d_all, I_df_u_all, F_all
    direct, downward diffuse, upward diffuse, actinic flux 
    (W/m^2, not W/m^2/m!)


"""

__all__ = ['available_schemes', 'res_keys_all_schemes']


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


