"""
Calculations/plots using the CRT solutions (irradiance spectral profiles),
using the model output dataset created by :meth:`crt1d.Model.to_xr()`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .spectra import _x_frac_in_bounds
from .spectra import BAND_DEFNS_UM
from .spectra import e_wl_umol


def _E_to_PFD_da(da):
    """Assuming wavelength in microns, convert W/m2 -> (μmol photons)/m2/s.
    Returns a new :class:`xr.DataArray`.
    """
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


def band(ds, *, variables=None, band_name="PAR", bounds=None, calc_PFD=False):
    """Reduce spectral variables in `ds` by summing in-band irradiances
    to give the integrated irradiance in a given spectral band.

    `bounds` does not have to be provided if `band_name` is one of the known bands.
    If `variables` is ``None``, we attempt to guess.

    Optionally, we additionally calculate photon flux density (PFD) variants of the quantities.
    It is important to do this at this stage (rather than after integrating) due to the non-linear
    dependence of photon energy on wavelength.

    Returns
    -------
    xr.Dataset
        New dataset with no wavelength dimension, only height.
    """
    ds = ds.copy()
    if bounds is None:
        bounds = BAND_DEFNS_UM[band_name]

    wl = ds.wl.values
    dwl = ds.dwl.values
    try:
        wle = ds.wle.values
    except AttributeError:
        # TODO: warn; use edges_from_centers instead?
        wle = np.r_[wl - 0.5 * dwl, wl[-1] + 0.5 * dwl[-1]]

    # weights as a function of wavelength
    w = xr.DataArray(dims="wl", data=_x_frac_in_bounds(wle, bounds))

    # default to searching for irradiance variables and actinic flux
    if variables is None:
        vns = [vn for vn in ds.variables if vn == "F" or "I" in vn]

    for vn in vns:
        da = ds[vn]
        ln_new = f"{da.attrs['long_name']} \u2013 {band_name}"
        units = da.attrs["units"]
        ds[vn] = (da * w).sum(dim="wl")
        ds[vn].attrs.update(
            {
                "long_name": ln_new,
                "units": units,
            }
        )
        if calc_PFD:
            da_pfd = _E_to_PFD_da(da)
            ln_new = f"{da_pfd.attrs['long_name']} \u2013 {band_name}"
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


def plot_compare_band(
    dsets,
    *,
    band_name="PAR",
    bounds=None,
    ref=None,
    ref_relative=False,
    marker=".",
    legend_outside=True,
    legend_labels="long_name",
):
    """Multi-panel plot of profiles for specified band.
    `bounds` does not have to be provided if `band_name` is one of the known bands.

    This uses :func:`band` to sum irradiances within the band.

    Usage examples: :ref:`examples/run-all-schemes:plots <examples/run-all-schemes:plots>`.

    Parameters
    ----------
    dsets : list of xr.Dataset
        Created using :meth:`~crt1d.Model.to_xr`.
    ref : str, xr.Dataset, optional
        Scheme ``name`` to be used as the reference
        (it will be subtracted from the others in the plot).
        Default: no reference used.
    ref_relative : bool
        Whether to present relative error when using a `ref`.
        If false, absolute error is presented.
    marker : str, optional
        Marker to be used when plotting the lines.
        Pass `None` to get no markers.
    legend_outside : bool
        Whether to place the legend outside the axes area or within.
        If outside, it will be placed upper right.
        If within, it will be placed on top of the lower left ax.
    legend_labels : {'long_name', 'short_name', 'name'}

    See Also
    --------
    band : Related function that returns a dataset for a given spectral band.
    """
    if not isinstance(dsets, list):
        raise Exception("A list of dsets must be provided")

    # Integrate over the band
    dsets = [band(ds, band_name=band_name, bounds=bounds) for ds in dsets]

    # Variables to show in each panel
    varnames = [
        ["aI", "aI_sl", "aI_sh"],
        ["I_dr", "I_df_d", "I_df_u"],
    ]  # rows, cols

    # Reference?
    if ref is not None:
        if isinstance(ref, str):
            names = [ds.scheme_name for ds in dsets]
            dsref = dsets[names.index(ref)]
        elif isinstance(ref, xr.Dataset):
            dsref = band(ref, band_name=band_name, bounds=bounds)
        else:
            raise TypeError("invalid type for `ref`.")
    else:
        dsref = None

    # Reference relative?
    if ref_relative:
        ref_sym = r"$\Delta_r$"
    else:
        ref_sym = r"$\Delta$"

    nrows = len(varnames)
    ncols = len(varnames[0])

    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(ncols * 2.4, nrows * 3))

    vns = [vn for row in varnames for vn in row]
    for i, vn in enumerate(vns):
        ax = axs.flat[i]
        for dset in dsets:
            da0 = dset[vn]
            if ref is not None:
                da = da0 - dsref[vn]
                if ref_relative:
                    da = da / dsref[vn]
                da.attrs.update(da0.attrs)
                da.attrs["long_name"] = f"{ref_sym} {da.long_name}"
            else:
                da = da0
            y = da.dims[0]
            da.plot(y=y, ax=ax, label=dset.attrs[f"scheme_{legend_labels}"], marker=marker)

    if ref is not None:
        axs.flat[0].set_title(f"Reference: {dsref.scheme_name}", loc="left", fontsize=10)

    for ax in axs.flat:
        ax.grid(True)
        ax.set_ylim(ymin=0)

    h, _ = axs.flat[0].get_legend_handles_labels()
    if legend_outside:
        # upper right outside axes
        fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.98, 0.987))
    else:
        # lower left inside ax
        fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.1, 0.13), fontsize=9)

    fig.tight_layout()


# TODO: def plot_E_closure_spectra():


def compare_ebal(dsets, *, band_name="solar", bounds=None):
    """
    For `dsets`, assess energy balance closure by comparing
    computed canopy and soil absorption to incoming minus outgoing
    radiation at the top of the canopy.

    Parameters
    ----------
    dsets : list of xr.Dataset
        Created using :meth:`~crt1d.Model.to_xr`.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(dsets, list):
        raise Exception("A list of dsets must be provided")

    # integrate over the band
    dsets = [band(ds, band_name=band_name, bounds=bounds) for ds in dsets]

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
