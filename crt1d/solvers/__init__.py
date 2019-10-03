"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary arguments using the cnpy_rad_state and cnpy_descrip 
dictionary inputs.


For now leaving all as looping over all available wavelengths. 

"""

from importlib import import_module
import inspect

__all__ = ['available_schemes', 'res_keys_all_schemes']

_solver_module_names = ['bl', 'twos_ds', 'fours', 'zq', 'bf', 'gd']
_scheme_IDs          = ['bl', '2s',      '4s',    'zq', 'bf', 'gd']
# available_schemes = [{} for _ in _solver_module_names]
available_schemes = { scheme_ID: {} for scheme_ID in _scheme_IDs }
# ^ OrderedDict ?

res_keys_all_schemes = ['I_dr', 'I_df_d', 'I_df_u', 'F']
# ^ the quantities all schemes must return. some return others as well


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

# TODO: delete the temporary variables from here
# or put the stuff in a function


