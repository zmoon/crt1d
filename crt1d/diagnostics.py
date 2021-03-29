"""
Calculations/plots from the CRT solutions, using the model's output dataset
created by :meth:`crt1d.Model.to_xr()`.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.constants import c
from scipy.constants import h
from scipy.constants import N_A


def e_wl_umol(wl_um):
    """J/(μmol photons) at wavelength `wl_um`."""
    # convert wavelength to SI units (for compatibility with the constants)
    wl = wl_um * 1e-6  # um -> m

    e_wl_1 = h * c / wl  # J (per one photon)
    e_wl_mol = e_wl_1 * N_A  # J/mol
    e_wl_umol = e_wl_mol * 1e-6  # J/mol -> J/umol

    return e_wl_umol


def E_to_PFD_da(da):
    """Assuming wl in microns, convert W/m2 -> (μmol photons)/m2/s."""
    assert da.attrs["units"] == "W m-2"
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
    """Weights for the computation of a band sum (integral).
    The calculation includes fractional contributions from bands that are partially
    within the `bounds`.

    Parameters
    ----------
    xe : array
        Edges
    bounds : 2-tuple(float)
        Bounds

    Raises
    ------
    UserWarning
        If the bounds extend outside the data range (ignored for solar).

    Returns
    -------
    array
        Weights, size ``xe.size - 1``.
    """
    x1, x2 = xe[:-1], xe[1:]  # left and right

    b1, b2 = bounds[0], bounds[1]

    if (b1 < x1[0] or b2 > x2[-1]) and bounds != BAND_DEFNS_UM["solar"]:
        warnings.warn(
            f"`bounds` ({b1:.3g}, {b2:.3g}) extend outside the data range "
            f"defined by `xe` ({x1[0]:.3g}, {x2[-1]:.3g})"
        )

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


def ds_band_sum(ds, *, variables=None, band_name="PAR", bounds=None, calc_PFD=False):
    """Reduce spectral variables in `ds` by summing in-band irradiances
    to give the integrated irradiance in that spectral region.
    `bounds` does not have to be provided if `band_name` is one of the known bands.
    If `variables` is None, will attempt to guess.
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

    # default to searching for irradiance variables and actinic flux
    if variables is None:
        vns = [vn for vn in ds.variables if vn == "F" or "I" in vn]

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


def plot_compare_band(dsets, *, band_name="PAR", bounds=None):
    """Multi-panel plot of profiles for specified string bandname `bn`.
    `bounds` does not have to be provided if `band_name` is one of the known bands.

    Parameters
    ----------
    dsets : list(xarray.Dataset)
        Created using :meth:`~crt1d.model.Model.to_xr`.
    """
    if not isinstance(dsets, list):
        raise Exception("A list of dsets must be provided")

    # integrate over the band
    dsets = [ds_band_sum(ds, band_name=band_name, bounds=bounds) for ds in dsets]

    # variables to show in each panel
    varnames = [
        [f"aI", f"aI_sl", f"aI_sh"],
        [f"I_dr", f"I_df_d", f"I_df_u"],
    ]  # rows, cols

    nrows = len(varnames)
    ncols = len(varnames[0])

    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(ncols * 2.4, nrows * 3))

    vns = [vn for row in varnames for vn in row]
    for i, vn in enumerate(vns):
        ax = axs.flat[i]
        for dset in dsets:
            da = dset[vn]
            y = da.dims[0]
            da.plot(y=y, ax=ax, label=dset.attrs["scheme_long_name"], marker=".")

    for ax in axs.flat:
        ax.grid(True)

    # legend in lower left
    h, _ = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.1, 0.13), fontsize=9)

    fig.tight_layout()


# TODO: def plot_E_closure_spectra():


def df_E_balance(dsets, *, band_name="solar", bounds=None):
    """
    For `dsets`, assess energy balance closure by comparing
    computed canopy and soil absorption to incoming minus outgoing
    radiation at the top of the canopy.

    Parameters
    ----------
    dsets : list(xarray.Dataset)
        :meth:`~crt1d.model.Model.to_xr`.

    """
    if not isinstance(dsets, list):
        raise Exception("A list of dsets must be provided")

    # integrate over the band
    dsets = [ds_band_sum(ds, band_name=band_name, bounds=bounds) for ds in dsets]

    IDs = [ds.attrs["scheme_name"] for ds in dsets]
    columns = [
        "incoming",
        "outgoing (reflected)",
        "soil absorbed",
        "layerwise abs sum",
        "in-out-soil",
        "canopy abs",
    ]
    df = pd.DataFrame(index=IDs, columns=columns)

    for i, ID in enumerate(IDs):
        ds = dsets[i]
        incoming = ds["I_d"][-1].values
        outgoing = ds["I_df_u"][-1].values
        transmitted = ds["I_d"][0].values  # direct+diffuse transmitted all the way through canopy
        soil_refl = ds["I_df_u"][0].values
        soil_abs = transmitted - soil_refl
        layer_abs_sum = ds["aI"].sum().values
        canopy_abs = (
            ds["I_df_d"][-1].values
            - outgoing
            + ds["I_dr"][-1].values
            - ds["I_dr"][0].values
            + -(ds["I_df_d"][0].values - soil_refl)
        )  # soil abs is down-up diffuse at last layer?
        df.loc[ID, columns[0]] = incoming
        df.loc[ID, columns[1]] = outgoing
        df.loc[ID, columns[2]] = soil_abs
        df.loc[ID, columns[3]] = layer_abs_sum
        df.loc[ID, columns[4]] = incoming - outgoing - soil_abs
        df.loc[ID, columns[5]] = canopy_abs

    return df
