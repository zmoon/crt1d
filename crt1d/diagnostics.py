"""
Calculations from the CRT solutions
"""
import warnings

import numpy as np
import xarray as xr
from scipy.constants import c
from scipy.constants import h
from scipy.constants import N_A

from .solvers import RET_KEYS_ALL_SCHEMES


def e_wl_umol(wl_um):
    """J/(umol photons) at wavelength `wl_um`."""
    # convert wavelength to SI units (for compatibility with the constants)
    wl = wl_um * 1e-6  # um -> m

    e_wl_1 = h * c / wl  # J (per one photon)
    e_wl_mol = e_wl_1 * N_A  # J/mol
    e_wl_umol = e_wl_mol * 1e-6  # J/mol -> J/umol

    return e_wl_umol


def E_to_PFD_da(da):
    """Assuming wl in microns, convert W/m2 -> (umol photons)/m2/s."""
    wl_um = da.wl
    f = 1 / e_wl_umol(wl_um)  # J/umol -> umol/J
    ln = da.attrs["long_name"].replace("irradiance", "PFD")
    units = "μmol photons m-2 s-1"
    da_new = da * f  # umol/J * W/m2 -> umol/m2/s
    da_new.attrs.update(
        {
            "long_name": ln,
            "units": units,
        }
    )

    return da_new


def band_sum_weights(xe, bounds):
    """Including fractional edge contributions.

    Parameters
    ----------
    xe : array
    bounds : 2-tuple(float)
    """
    x1, x2 = xe[:-1], xe[1:]  # left and right

    b1, b2 = bounds[0], bounds[1]

    if b1 < x1[0] or b2 > x2[-1]:
        warnings.warn("`bounds` extend outside the data range")

    in_bounds = (x2 >= b1) & (x1 <= b2)
    x1in, x2in = x1[in_bounds], x2[in_bounds]
    dx = x2in - x1in
    # db = b2 - b1

    # need to find a better (non-loop) way to do this, but for now...
    w = np.zeros_like(x1in)
    for i in range(x1in.size):
        x1in_i, x2in_i = x1in[i], x2in[i]
        if x1in_i < b1:  # extending out on left
            w[i] = (x2in_i - b1) / dx[i]
        elif x2in_i > b2:  # extending out on right
            w[i] = (b2 - x1in_i) / dx[i]
        else:  # completely within the bounds
            w[i] = 1

    w_all = np.zeros(xe.size - 1)
    w_all[in_bounds] = w

    return w_all


def calc_band_sum(y, xe, bounds):
    assert y.size + 1 == xe.size, "Edges must have one more value than `y`."
    w = band_sum_weights(xe, bounds)

    return np.sum(y * w)


BAND_DEFNS_UM = {
    "PAR": (0.4, 0.7),
    "NIR": (0.7, 2.5),
    "UV": (0.01, 0.4),
    "solar": (0.3, 5.0),
}


def ds_band_sum(ds, *, band_name="PAR", bounds=None, calc_PFD=False):
    """Reduce spectral variables in `ds` by summing in-band irradiances.
    `bounds` does not have to be provided if `band_name` is one of the known bands.
    """
    ds = ds.copy()
    if bounds is None:
        bounds = BAND_DEFNS_UM[band_name]

    wl = ds.wl.values
    dwl = ds.dwl.values
    try:
        wle = ds.wle.values
    except AttributeError:
        wle = np.r_[wl - 0.5 * dwl, wl[-1] + 0.5 * dwl[-1]]

    # weights as a function of wavelength
    w = xr.DataArray(dims="wl", data=band_sum_weights(wle, bounds))

    vns = [vn for vn in ds.variables if vn not in ds.coords and "wl" in ds[vn].coords]
    for vn in vns:
        da = ds[vn]
        ln_new = f"{da.attrs['long_name']} - {band_name}"
        units = da.attrs["units"]
        ds[vn] = (da * w).sum(dim="wl")
        ds[vn].attrs.update(
            {
                "long_name": ln_new,
                "units": units,
            }
        )
        if calc_PFD:
            da_pfd = E_to_PFD_da(da)
            ln_new = f"{da_pfd.attrs['long_name']} - {band_name}"
            units = da_pfd.attrs["units"]
            vn_pfd = vn.replace("I", "PFD")
            ds[vn_pfd] = (da_pfd * w).sum(dim="wl")
            ds[vn_pfd].attrs.update(
                {
                    "long_name": ln_new,
                    "units": units,
                }
            )

    ds.attrs.update(
        {
            "band_name": band_name,
            "band_bounds": bounds,
        }
    )

    return ds.drop("wl")


# TODO: maybe want to create and return as xr.Dataset instead
# or like sympl, dict of xr.Dataarrays
def calc_leaf_absorption(p, out, *, band_names_to_calc=None):
    """Calculate layerwise leaf absorption from the CRT solver solutions

    Parameters
    ----------
    p : dict
        model parameters
    out : dict
        model outputs

    for the state (one time/case)
    """
    if band_names_to_calc is None:
        band_names_to_calc = ["PAR", "solar"]
    elif band_names_to_calc == "all":
        band_names_to_calc = ["PAR", "solar", "NIR", "UV"]

    lai = p["lai"]
    dlai = p["dlai"]
    # K_b = p["K_b"]
    # G = p["G"]  # fractional leaf area projected in direction psi
    K_b = p["K_b"]  # G/cos(psi)

    leaf_r = p["leaf_r"]
    leaf_t = p["leaf_t"]
    leaf_a = 1 - (leaf_r + leaf_t)  # leaf element absorption coeff

    wl = p["wl"]
    # dwl = p["dwl"]
    I_dr = out["I_dr"]
    I_df_d = out["I_df_d"]
    I_df_u = out["I_df_u"]
    I_d = I_dr + I_df_d  # direct+diffuse downward irradiance

    # get band wavelength bounds, in order to assess contributions to the integrals
    try:
        wl_l = p["wl_l"]
        wl_r = p["wl_r"]
    except KeyError:
        warnings.warn(
            "Wavelength band left and right bounds should be provided "
            "to give the most accurate integrals."
        )
        wl_l = wl  # just use the wavelength we have for defining the bands
        wl_r = wl

    # TODO: include clump factor in f_sl and absorption calculations

    f_sl_interfaces = np.exp(-K_b * lai)  # fraction of sunlit
    f_sl = f_sl_interfaces[:-1] + 0.5 * np.diff(f_sl_interfaces)
    f_sh = 1 - f_sl

    nlev = lai.size

    # compute total layerwise absorbed (by leaves, but not per unit LAI)
    i = np.arange(nlev - 1)
    ip1 = i + 1
    a = I_dr[ip1] - I_dr[i] + I_df_d[ip1] - I_df_d[i] + I_df_u[i] - I_df_u[ip1]
    # ^ layerwise absorption by leaves in all bands
    #   inputs - outputs
    #   actual, not per unit LAI

    # compute absorbed direct
    I_dr0 = I_dr[-1, :][np.newaxis, :]
    # a_dr =  I_dr0 * (K_b*f_sl*dlai)[:,np.newaxis] * leaf_a
    a_dr = I_dr0 * (1 - np.exp(-K_b * f_sl * dlai))[:, np.newaxis] * leaf_a
    # ^ direct beam absorption (sunlit leaves only by definition)
    #
    # **technically should be computed with exp**
    #   1 - exp(-K_b*L)
    # K_b*L is an approximation for small L
    # e.g.,
    #   K=1, L=0.01, 1-exp(-K*L)=0.00995

    a_df = a - a_dr  # absorbed diffuse radiation
    a_df_sl = a_df * f_sl[:, np.newaxis]
    a_df_sh = a_df * f_sh[:, np.newaxis]
    a_sl = a_df_sl + a_dr
    a_sh = a_df_sh
    assert np.allclose(a_sl + a_sh, a)

    # compute spectral absorption profile in photon flux density units
    # a_pfd = E_to_PFD(a, wl)

    # > absorbance in specific bands
    isPAR = (wl_l >= 0.4) & (wl_r <= 0.7)
    isNIR = (wl_l >= 0.7) & (wl_r <= 2.5)
    isUV = (wl_l >= 0.01) & (wl_r <= 0.4)
    issolar = (wl_l >= 0.3) & (wl_r <= 5.0)

    # TODO: fractional weights for bands that are only partially in the desired range

    # integrate spectral profiles over different bands (currently hardcoded)
    #
    # for the spectral version (/dwl), we can use an integral
    # a_PAR = si.simps(Sa[:,isPAR], wl[isPAR])
    #
    # for non-spectral, we just need to sum the bands
    # a_PAR = a[:,isPAR].sum(axis=1)
    #
    bands = {
        "PAR": isPAR,
        "solar": issolar,
        "UV": isUV,
        "NIR": isNIR,
    }
    to_int = {
        "aI": a,
        "aI_df": a_df,
        "aI_dr": a_dr,
        "aI_sh": a_sh,
        "aI_sl": a_sl,
        "aI_df_sl": a_df_sl,
        "aI_df_sh": a_df_sh,
        "I_d": I_d,
        "I_dr": I_dr,
        "I_df_d": I_df_d,
        "I_df_u": I_df_u,
    }
    res_int = {}  # store integration results in dict
    for i, bn in enumerate(band_names_to_calc):
        isband = bands[bn]
        for k, v in to_int.items():
            k_int = f"{k}_{bn}"  # e.g., `aI_PAR`
            res_int[k_int] = v[:, isband].sum(axis=1)
            new_base = k.replace("I", "PFD")  # TODO: don't want to be replacing NIR though!
            k_int_pfd = f"{new_base}_{bn}"
            res_int[k_int_pfd] = E_to_PFD(v[:, isband], wl[isband]).sum(axis=1)

    to_int_new = {
        k: v for k, v in to_int.items() if k not in RET_KEYS_ALL_SCHEMES
    }  # this is the new spectra
    res = {**to_int_new, **res_int}  # new spectra + integrated quantities

    return res


# TODO: fn to parse string varname to get dims/coords, units, and long_name
# need to move from model.to_xr and write better way (re or pyparsing?)
