"""
1-D canopy radiative transfer
"""
from pathlib import Path as _Path

# directories
BASE_DIR = _Path(__file__).parent
DATA_BASE_DIR = BASE_DIR / "data"

# include Model in pkg-level namespace
from .model import Model  # noqa: F401 unused import

# include diagnostics module (not used by any of the others)
from . import diagnostics  # noqa: F401 unused import

# set version
try:
    from . import _version

    __version__ = _version.version
except ModuleNotFoundError:
    # the package is probably not installed
    pass


def print_config():
    """Print info about the schemes etc."""
    from .solvers import AVAILABLE_SCHEMES

    # config summary
    sconfig = f"""
scheme IDs available: {', '.join(AVAILABLE_SCHEMES.keys())}
crt1d base dir
    {BASE_DIR.as_posix():s}
looking for input data for provided cases in
    {DATA_BASE_DIR.as_posix():s}
    """.strip()
    print(sconfig)
