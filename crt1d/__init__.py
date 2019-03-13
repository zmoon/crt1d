#!/usr/bin/env python3
__author__ = "Z Moon"
__copyright__ = ""
__version__ = ""
__license__ = "GPLv3"
__email__ = "zlm1@psu.edu"

#from .spectral_library import get_spectra
#spectral_lib = get_spectra()

from .solvers import available_schemes
print('scheme IDs available: {:s}'.format(' '.join([as_['ID'] for as_ in available_schemes])))

import os
crt1d_base_dir = os.path.dirname(os.path.realpath(__file__))
input_data_dir = '{:s}/data'.format(crt1d_base_dir)
print('looking for necessary model input data in\n    {:s}'.format(input_data_dir))

from .crt1d import model
