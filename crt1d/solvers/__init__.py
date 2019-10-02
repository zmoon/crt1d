"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary arguments using the cnpy_rad_state and cnpy_descrip 
dictionary inputs.


For now leaving all as looping over all available wavelengths. 

"""

from importlib import import_module
import inspect


import numpy as np
import scipy.integrate as si
import scipy.optimize as so

# from .bl import solve_bl
# from .twos_ds import solve_2s
# from .fours import solve_4s
# from .zq import solve_zq
# from .bf import solve_bf
# from .gd import solve_gd

__all__ = ['available_schemes']


# available_schemes = [\
#     dict(ID='bl', solver=solve_bl, 
#         shortname='B–L', longname='Beer–Lambert'),
#     dict(ID='2s', solver=solve_2s, 
#         shortname='2s', longname='Dickinson–Sellers two-stream'),
#     dict(ID='4s', solver=solve_4s, 
#         shortname='4s', longname='four-stream'),
#     dict(ID='zq', solver=solve_zq, 
#         shortname='ZQ', longname='Zhao & Qualls multi-scattering'),
#     dict(ID='bf', solver=solve_bf, 
#         shortname='BF', longname='Bodin & Franklin improved Goudriaan'),
#     dict(ID='gd', solver=solve_gd, 
#         shortname='Gou', longname='Goudriaan'),
#     ]

_solver_module_names = ['bl', 'twos_ds', 'fours', 'zq', 'bf', 'gd']
_scheme_IDs          = ['bl', '2s',      '4s',    'zq', 'bf', 'gd']
# available_schemes = [{} for _ in _solver_module_names]
available_schemes = { scheme_ID: {} for scheme_ID in _scheme_IDs }
# ^ OrderedDict ?

# for name, scheme_ID, scheme_dict in zip(_solver_module_names, _scheme_IDs, available_schemes):
for name, (scheme_ID, scheme_dict) in zip(_solver_module_names, available_schemes.items()):

    scheme_dict['module_name'] = name
    scheme_dict['ID'] = scheme_ID

    # import things from the individual files
    module = import_module(f'.{name}', package='crt1d.solvers')
    # print(module)
    # print(dir(module))
    solve_fun_name = f'solve_{scheme_ID}'  # or could change the fn names to just be 'solve' and import alias here
    # solver = module.__get__(solve_fun_name)
    solver = getattr(module, solve_fun_name)
    # globals.update()  # add solver function to namespace?
    # try:
    #     short_name = module.short_name
    #     long_name = module.long_name
    # except AttributeError:
    #     short_name = ''
    #     long_name = ''
    short_name = getattr(module, 'short_name', '')
    long_name = getattr(module, 'long_name', '')

    scheme_dict['short_name'] = short_name
    scheme_dict['long_name'] = long_name
    scheme_dict['solver'] = solver


for scheme_dict in available_schemes.values():
    fullargspec = inspect.getfullargspec(scheme_dict['solver'])
    # scheme_dict['args'] = fullargspec.args
    scheme_dict['args'] = fullargspec.kwonlyargs



def e_to_photon(E, wl_um):
    """Energy flux (W/m^2) to photon flux (umol photon / m^2 / s) ?
    """
    h = 6.626e-34
    c = 3.0e8
    N_A = 6.022e23

    wl = wl_um * 1e-6  # um -> m
    
    e_wl_1 = h*c/wl
    e_wl_mol = e_wl_1 * N_A

    return e_wl_mol * 1e6  # mol -> umol



def calc_leaf_absorption(cd, crs):
    """Calculate layerwise leaf absorption from the CRT solver solutions"""

    lai = cd['lai']
    dlai = cd['dlai']
    K_b = crs['K_b']
    G = crs['G']  # fractional leaf area projected in direction psi
    K_b = crs['K_b']  # G/cos(psi)

    leaf_r = cd['leaf_r']
    leaf_t = cd['leaf_t']
    leaf_a = 1 - (leaf_r + leaf_t)  # leaf element absorption coeff

    wl     = crs['wl']
    dwl    = crs['dwl']
    I_dr   = crs['I_dr']
    I_df_d = crs['I_df_d']
    I_df_u = crs['I_df_u']
    I_d = I_dr + I_df_d  # downward irradiance

    f_sl_interfaces = np.exp(-K_b*lai)  # fraction of sunlit
    f_sl = f_sl_interfaces[:-1] + 0.5*np.diff(f_sl_interfaces)
    f_sh = 1 - f_sl

    nlev = lai.size

    i = np.arange(nlev-1)
    ip1 = i + 1
    a = I_dr[ip1]-I_dr[i] + I_df_d[ip1]-I_df_d[i] + \
        I_df_u[i]-I_df_u[ip1]
    # ^ layerwise absorption by leaves in all bands
    #   inputs - outputs
    #   actual, not per unit LAI

    I_dr0 = I_dr[-1,:][np.newaxis,:]
    a_dr_sl =  I_dr0 * (K_b*f_sl*dlai)[:,np.newaxis] * leaf_a
    a_sl = (a-a_dr_sl)*f_sl[:,np.newaxis] + a_dr_sl
    a_sh = (a-a_dr_sl)*f_sh[:,np.newaxis]
    assert( np.allclose(a_sl+a_sh, a) )

    # #> absorbance in specific bands
    isPAR = (wl >= 0.4) & (wl <= 0.7)
    # isNIR = (wl >= 0.7) & (wl <= 2.5)
    # isUV  = (wl >= 0.01) & (wl <= 0.4)

    a_PAR = si.simps(a[:,isPAR], wl[isPAR])
    # a_NIR = si.simps(a[isNIR], wl[isNIR])
    # a_UV  = si.simps(a[isUV],  wl[isUV])

    res = {
        'aI' : a,
        'aI_sl': a_sl, 
        'aI_sh': a_sh,
        'aI_PAR': a_PAR

    }

    return res








