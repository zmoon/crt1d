#!/usr/bin/env python3
__author__ = "Z Moon"
__copyright__ = ""
__version__ = ""
__license__ = "GPLv3"
__email__ = "zlm1@psu.edu"

#from .spectral_library import get_spectra
#spectral_lib = get_spectra()

from .solvers import available_schemes
# print( ))

import os
crt1d_base_dir = os.path.dirname(os.path.realpath(__file__))
input_data_dir = '{:s}/data'.format(crt1d_base_dir)
# print()

from .crt1d import model

# print('test')

sconfig = f"""
scheme IDs available: {', '.join(available_schemes.keys())}
crt1d base dir
  {crt1d_base_dir}
looking for input data for provided cases in
  {input_data_dir}
""".strip()

#@property
def config():
    print(sconfig)
