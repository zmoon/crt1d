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








