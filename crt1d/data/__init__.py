"""
Data provided with crt1d: default/sample spectra,
as well as functions that use external (not-required) libraries to compute more spectra.

Data are loaded as :class:`xarray.Dataset` for easy inspection.
"""
import numpy as np
import xarray as xr

from .. import DATA_BASE_DIR
from ..spectra import edges_from_centers
from ..variables import _tup
from .external import solar_sp2

DATA_DIR_STR = DATA_BASE_DIR.as_posix()


def _wl_coord_dict(wl, *, units="μm"):
    return {"wl": (("wl"), wl, {"long_name": "Wavelength", "units": units})}


def load_soil_fuentes2007(wl_um):
    """PAR and NIR values, applied to the wavelengths in `wl_um`."""
    wl_um = np.asarray(wl_um)
    r_soil = np.ones_like(wl_um)
    r_soil[wl_um <= 0.7] = 0.1100  # this is the PAR value
    r_soil[wl_um > 0.7] = 0.2250  # near-IR value

    attrs = {}
    return xr.Dataset(
        coords=_wl_coord_dict(wl_um),
        data_vars={"rs": _tup("soil_r", r_soil)},
        attrs=attrs,
    )


def load_prosail_sample_soil(*, f_wet=0):
    """Sample soil spectra provided in Python PROSAIL.

    Parameters
    ----------
    f_wet : float
        Weighting factor for the wet soil spectrum (0--1)

    Notes
    -----
    https://github.com/jgomezdans/prosail/blob/master/prosail/soil_reflectance.txt
    """
    rho_soil_dry, rho_soil_wet = np.loadtxt(DATA_BASE_DIR / "PROSAIL_sample-soil.txt", unpack=True)
    wl_ps5 = load_default_ps5()["wl"]

    # weighted sum of wet and dry soil spectra
    assert f_wet >= 0 and f_wet <= 1, "`f_wet` must be in [0, 1]"
    rho_soil = f_wet * rho_soil_wet + (1 - f_wet) * rho_soil_dry

    attrs = {}
    dset = xr.Dataset(
        coords={
            "wl": wl_ps5,
        },
        data_vars={
            "rs": _tup("soil_r", rho_soil),
        },
        attrs=attrs,
    )
    return dset


def load_default_ps5():
    """Load the provided default PROSPECT5 leaf element reflectance/transmittance.

    (I believe) these are the default spectra for the online version of PROSPECT.
    """
    wl_nm, r, t = np.loadtxt(DATA_BASE_DIR / "PROSPECT_sample.txt", unpack=True)
    wl = wl_nm / 1000.0  # nm->um

    attrs = {}
    ds = xr.Dataset(
        coords=_wl_coord_dict(wl),
        data_vars={
            "rl": _tup("leaf_r", r),
            "tl": _tup("leaf_t", t),
        },
        attrs=attrs,
    )
    return ds


def load_ideal_leaf(*, midpt=False):
    """Load the ideal green leaf properties (at SPCTRAL2 wavelengths)."""
    fpath = DATA_BASE_DIR / "ideal-green-leaf_SPCTRAL2-wavelengths.csv"
    wl, t, r = np.loadtxt(fpath, delimiter=",", skiprows=1, unpack=True)

    # adjust <= 0 values (on the right edge where r=t=0)
    t[t == 0] = 1e-10
    r[r == 0] = 1e-10

    if midpt:
        wl = (wl[:-1] + wl[1:]) / 2
        t = (t[:-1] + t[1:]) / 2
        r = (r[:-1] + r[1:]) / 2

    attrs = {}
    ds = xr.Dataset(
        coords=_wl_coord_dict(wl),
        data_vars={
            "rl": _tup("leaf_r", r),
            "tl": _tup("leaf_t", t),
        },
        attrs=attrs,
    )
    return ds


def load_default_sp2(*, midpt=True):
    """Load sample SPCTRAL2 top-of-canopy spectra (direct and diffuse).

    This is the default spectrum in the Excel version of SPCTRAL2.

    Parameters
    ----------
    midpt : bool, optional
        true (default): irradiance calculated at midpts of the original grid

        false: irradiance calculated by estimating the edges of the original grid
        and treating the original grid as centers

    Notes
    -----
    The SPCTRAL2 irradiances are in spectral form: W m-2 μm-1,
    but we need in-band irradiance for the solvers, since some compute W/m2 absorption.
    """
    # load original spectra
    fp = DATA_BASE_DIR / "SPCTRAL2_xls_default-spectrum.csv"
    wl0, SI_dr0, SI_df0 = np.loadtxt(fp, delimiter=",", skiprows=1, unpack=True)

    if midpt:
        wle = wl0
        dwl = np.diff(wle)
        wl = wl0[:-1] + 0.5 * dwl  # for in-band irradiance

        # spectral irradiance for bin center as average of edge values
        # integration: spectral irradiance -> bin-level irradiance
        I_dr = (SI_dr0[:-1] + SI_dr0[1:]) / 2 * dwl
        I_df = (SI_df0[:-1] + SI_df0[1:]) / 2 * dwl

    else:  # edges-from-centers method
        wle = edges_from_centers(wl0)
        dwl = np.diff(wle)
        wl = wl0  # for in-band irradiance (and original spectral irradiance)

        I_dr = SI_dr0 * dwl
        I_df = SI_df0 * dwl

    dimsx = "wl"  # band edges / actual spectrum tracing
    dims0 = "wl0"  # band centers
    Sunits = "W m-2 μm-1"  # spectral units
    units = "W m-2"  # non-spectral

    attrs = {}
    # TODO: use `_tup`
    coords = {
        "wl": (dimsx, wl, {"units": "μm", "long_name": "Wavelength"}),
        "wl0": (dims0, wl0, {"units": "μm", "long_name": "Wavelength in original spectra"}),
        "wle": ("wle", wle, {"units": "μm", "long_name": "Irradiance band edge wavelength"}),
    }
    dset = xr.Dataset(
        coords=coords,
        data_vars={
            "SI_dr": (dims0, SI_dr0, {"units": Sunits, "long_name": "Spectral direct irradiance"}),
            "SI_df": (
                dims0,
                SI_df0,
                {"units": Sunits, "long_name": "Spectral diffuse irradiance"},
            ),
            "I_dr": (dimsx, I_dr, {"units": units, "long_name": "Direct irradiance"}),
            "I_df": (dimsx, I_df, {"units": units, "long_name": "Diffuse irradiance"}),
            "dwl": (dimsx, dwl, {"units": "μm", "long_name": "Wavelength band width"}),
        },
        attrs=attrs,
    )
    return dset


def load_default(*, midpt=True):
    """Load default toc irradiance spectra, and leaf & soil optical props,
    used in :func:`~crt1d.cases.load_default_case`.
    """
    # load individual datasets
    ds_l = load_ideal_leaf(midpt=midpt)
    ds_r = load_default_sp2(midpt=midpt)  # toc irradiance
    ds_s = load_soil_fuentes2007(wl_um=ds_l.wl)  # soil, using the leaf wavelengths

    # make sure the grids are the same
    assert np.allclose(ds_l.wl, ds_s.wl)
    assert np.allclose(ds_l.wl, ds_r.wl)

    if not midpt:
        assert np.allclose(
            ds_l.wl, ds_r.wl0
        ), "In the non-midpt case, we use the original SPCTRAL2 wavelengths."

    # join into one dataset
    ds = xr.merge([ds_l, ds_r, ds_s])

    return ds
