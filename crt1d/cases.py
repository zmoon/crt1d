"""
(Complete) sets of input parameters.
"""
import numpy as np

from . import data
from . import DATA_BASE_DIR
from .leaf_angle import G_ellipsoidal_approx
from .leaf_angle import mla_to_x_approx as mla_to_orient
from .leaf_area import distribute_lai_beta
from .leaf_area import distribute_lai_from_cdd


input_data_dir_str = DATA_BASE_DIR.as_posix()  # string


def load_default_case(nlayers):
    """Idealized beta leaf dist, default spectra."""
    h_c = 20.0
    LAI = 4.0
    res = distribute_lai_beta(h_c, LAI, nlayers)
    lai, z = res.lai, res.z

    # default spectral things,
    # dropping wavelengths where something is NaN (missing / not defined)
    ds = data.load_default(midpt=True).dropna(dim="wl")

    mla = 57  # for spherical? (check)
    orient = mla_to_orient(mla)
    G_fn = lambda psi_: G_ellipsoidal_approx(psi_, orient)

    sza = 20  # deg.
    psi = np.deg2rad(sza)
    # mu = np.cos(psi)

    p = dict(
        lai=lai,
        z=z,
        green=1.0,
        mla=mla,
        clump=1.0,
        orient=orient,
        G_fn=G_fn,
        psi=psi,
        #
        leaf_t=ds.tl.values,
        leaf_r=ds.rl.values,
        soil_r=ds.rs.values,
        wl_leafsoil=ds.wl.values,
        #
        I_dr0_all=ds.I_dr.values,
        I_df0_all=ds.I_df.values,
        wl=ds.wl.values,
        dwl=ds.dwl.values,
    )

    return p


def load_canopy_descrip(fpath):
    """Load items from CSV file and return as dict
    The file should use the same fieldnames as in the default one!
    Pandas might also do this.
    """
    varnames, vals = np.genfromtxt(
        fpath, usecols=(0, 1), skip_header=1, delimiter=",", dtype=str, unpack=True
    )

    n = varnames.size
    d_raw = {varnames[i]: vals[i] for i in range(n)}

    d = d_raw  # could copy instead
    for k, v in d.items():
        if ";" in v:
            d[k] = [float(x) for x in v.split(";")]
        else:
            d[k] = float(v)

    # checks
    assert np.array(d["lai_frac"]).sum() == 1.0

    return d


def load_Borden95_default_case(nlayers):
    """ """
    cdd_default = load_canopy_descrip(DATA_BASE_DIR / "default_canopy_descrip.csv")
    lai, z = distribute_lai_from_cdd(cdd_default, nlayers)

    raise NotImplementedError  # TODO: finish this
