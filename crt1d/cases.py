"""
(Complete) sets of input parameters.
"""
import numpy as np

from . import DATA_BASE_DIR
from .data import load_default_leaf_soil_props
from .data import load_default_toc_spectra
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

    # spectral things
    # includes:
    # - top-of-canopy BC, leaf reflectivity and trans., soil refl,
    #   for now, assume wavelengths from the BC apply to leaf as well
    (
        wl_default,
        dwl_default,
        leaf_r_default,
        leaf_t_default,
        soil_r_default,
    ) = load_default_leaf_soil_props()

    mla = 57  # for spherical? (check)
    orient = mla_to_orient(mla)
    G_fn = lambda psi_: G_ellipsoidal_approx(psi_, orient)

    cnpy_descrip = dict(
        lai=lai,
        z=z,
        green=1.0,
        mla=mla,
        clump=1.0,
        leaf_t=leaf_t_default,
        leaf_r=leaf_r_default,
        soil_r=soil_r_default,
        wl_leafsoil=wl_default,
        orient=orient,
        G_fn=G_fn,
    )

    wl, dwl, I_dr0, I_df0 = load_default_toc_spectra()

    sza = 20  # deg.
    psi = np.deg2rad(sza)
    # mu = np.cos(psi)

    cnpy_rad_state = dict(
        I_dr0_all=I_dr0,
        I_df0_all=I_df0,
        wl=wl,
        dwl=dwl,
        psi=psi,
    )

    return {**cnpy_descrip, **cnpy_rad_state}  # now combined


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
    cdd_default = load_canopy_descrip(input_data_dir_str + "/" + "default_canopy_descrip.csv")
    lai, z = distribute_lai_from_cdd(cdd_default, nlayers)

    raise NotImplementedError  # TODO: finish this
