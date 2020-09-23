"""
1-D canopy radiative transfer
"""
import os

# directories TODO: use pathlib
crt1d_base_dir = os.path.dirname(os.path.realpath(__file__))
input_data_dir = '{:s}/data'.format(crt1d_base_dir)

from .model import Model
from .solvers import available_schemes
from . import _version

# set version
try:
    __version__ = _version.version
except PackageNotFoundError:
    # package is probably not installed
   pass

#from .spectral_library import get_spectra
#spectral_lib = get_spectra()

# config summary
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
