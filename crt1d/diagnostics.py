"""
Calculations/plots using the CRT solutions (irradiance spectral profiles),
using the model output dataset created by :meth:`crt1d.Model.to_xr()`.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .spectra import _x_frac_in_bounds
from .spectra import BAND_DEFNS_UM
from .spectra import e_wl_umol
from .utils import cf_units_to_tex as _cf_units_to_tex


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
        warnings.warn(
            "`wle` was not present so we are computing the wave band edges "
            "from the band centers (`wl`) and band widths (`dwl`)."
        )
        wle = np.r_[wl[0] - 0.5 * dwl[0], wl + 0.5 * dwl]

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
    marker=".",
    ref=None,
    ref_label=None,
    ref_relative=False,
    ds_labels="scheme_long_name",
    legend_outside=True,
):
    """Multi-panel plot to compare profiles for the specified band for a set of datasets.
    `bounds` does not have to be provided if `band_name` is one of the known bands.

    This uses :func:`band` to sum irradiances within the band.

    :ref:`Usage examples <examples/run-all-schemes:plots -- single band>`

    Parameters
    ----------
    dsets : list of xr.Dataset
        Created using :meth:`crt1d.Model.to_xr`.
    marker : str, optional
        Marker to be used when plotting the lines.
        Pass `None` to get no markers
        (this can make it easier to see what's going on if there are many levels).
    ref : str, xr.Dataset, optional
        Dataset to be used as the reference
        (it will be subtracted from the others in the plot).
        If ``str``, the first found with ``name`` `ref` will be used,
        so ``str`` should only be used when comparing disparate schemes, like ``2s`` vs ``4s``).
        Default: no reference.
    ref_label : str, optional
        If not provided, a dataset attribute will be used
        (``scheme_short_name`` or the one being used for legend labels).
    ref_relative : bool, optional
        Whether to present relative error when using a `ref`.
        If false (default), absolute error is presented.
    ds_labels : str or list of str, optional
        If ``str``, must be a dataset attribute (that all in `dsets` have).
        If ``list``, should be same length as `dsets`.
    legend_outside : bool, optional
        Whether to place the legend outside the axes area or within.
        If outside, it will be placed upper right.
        If within, it will be placed on top of the lower left ax.

    See Also
    --------
    band : Related function that returns a dataset for a given spectral band.
    plot_compare_spectra :
        Similar routine, but plots one spectral variable as function of height and wavelength.
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

    # Dataset labels
    if isinstance(ds_labels, str):
        labels = [ds.attrs[ds_labels] for ds in dsets]
        if ref and ref_label is None:
            ref_label = dsref.attrs[ds_labels]
    elif isinstance(ds_labels, list):
        labels = ds_labels[:]
        if ref and ref_label is None:
            iref = None
            for i, ds in enumerate(dsets):
                if ds is dsref:
                    iref = i
                    break
            ref_label = labels[i] if iref is not None else dsref.attrs["scheme_short_name"]
    else:
        raise TypeError
    assert len(labels) == len(dsets)

    nrows = len(varnames)
    ncols = len(varnames[0])

    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(ncols * 2.4, nrows * 3))

    vns = [vn for row in varnames for vn in row]
    for i, vn in enumerate(vns):
        ax = axs.flat[i]
        for label, dset in zip(labels, dsets):
            da0 = dset[vn]

            if ref is not None:
                da = da0 - dsref[vn]
                if ref_relative:
                    da = da / dsref[vn]
                da.attrs["long_name"] = f"{ref_sym} {da0.long_name}"
                da.attrs["units"] = da0.units if not ref_relative else ""
            else:
                da = da0.copy()

            da.attrs["units"] = _cf_units_to_tex(da.units)

            da.plot(y=da.dims[0], ax=ax, label=label, marker=marker)

        if i % ncols:  # not in first column
            ax.set_ylabel("")

    if ref is not None:
        axs.flat[0].set_title(f"Reference: {ref_label}", loc="left", fontsize=10)

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


def plot_compare_spectra(
    dsets,
    which="I_d",
    *,
    dwl_relative=True,
    toc_relative=False,
    ref=None,
    ref_label=None,
    ref_relative=False,
    ref_plot=False,
    ds_labels="scheme_short_name",
    norm=None,
    plot_type="pcolormesh",
):
    """Multi-panel plot to compare spectra at each height for a set of datasets.

    :ref:`Usage examples <examples/run-all-schemes:plots -- spectra>`

    Parameters
    ----------
    dsets : list of xr.Dataset
        Created using :meth:`crt1d.Model.to_xr`.
    which : str
        Which spectrum to compare.
        Default: ``"I_d"`` (downward irradiance).
    dwl_relative : bool, optional
        Whether to divide by the wavelength band width.
        Default: true.
    toc_relative : bool, optional
        Whether to divide by the top-of-canopy spectrum.
    ref : str, xr.Dataset, optional
        Dataset to be used as the reference
        (it will be subtracted from the others in the plot).
        If ``str``, the first found with ``name`` `ref` will be used,
        so ``str`` should only be used when comparing disparate schemes, like ``2s`` vs ``4s``).
        Default: no reference.
    ref_label : str, optional
        If not provided, a dataset attribute will be used
        (``scheme_short_name`` or the one being used for legend labels).
    ref_relative : bool, optional
        Whether to present relative error when using a `ref`.
        If false (default), absolute error is presented.
    ref_plot : bool, optional
        Whether to plot the reference.
    ds_labels : str or list of str, optional
        If ``str``, must be a dataset attribute (that all in `dsets` have).
        If ``list``, should be same length as `dsets`.
    norm : matplotlib.colors.Normalize, optional
        Used if provided.

    See Also
    --------
    plot_compare_band :
        Similar routine, but integrates over a wave band and plots profiles of multiple variables.
    """
    # TODO: reduce duplicated code between this and `plot_compare_band`

    if plot_type not in ["pcolormesh", "contourf"]:
        raise ValueError
    if plot_type != "pcolormesh":
        raise NotImplementedError

    # Reference?
    if ref is not None:
        if isinstance(ref, str):
            names = [ds.scheme_name for ds in dsets]
            dsref = dsets[names.index(ref)]
        elif isinstance(ref, xr.Dataset):
            dsref = ref
        else:
            raise TypeError("invalid type for `ref`.")
    else:
        dsref = None

    # Reference relative?
    if ref_relative:
        ref_sym = r"$\Delta_r$"
    else:
        ref_sym = r"$\Delta$"

    # Dataset labels
    if isinstance(ds_labels, str):
        labels = [ds.attrs[ds_labels] for ds in dsets]
        if ref and ref_label is None:
            ref_label = dsref.attrs[ds_labels]
    elif isinstance(ds_labels, list):
        labels = ds_labels[:]
        if ref and ref_label is None:
            iref = None
            for i, ds in enumerate(dsets):
                if ds is dsref:
                    iref = i
                    break
            ref_label = labels[i] if iref is not None else dsref.attrs["scheme_short_name"]
    else:
        raise TypeError
    assert len(labels) == len(dsets)

    ncols = np.ceil(np.sqrt(len(dsets))).astype(int)
    nrows = np.ceil(len(dsets) / ncols).astype(int)

    figw, figh = ncols * 2.8, nrows * 2.2
    if ref_plot:
        figw += 1.7
    fig, axs = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=True,
        figsize=(figw, figh),
        gridspec_kw=dict(hspace=0.025, wspace=0.03),
    )

    imr = None
    ims = []
    for label, ds, ax in zip(labels, dsets, axs.flat):
        da = ds[which]
        ln = da.long_name
        un = da.units

        dar = dsref[which] if dsref else None

        # TODO: maybe should be default
        if dwl_relative:  # conversion to spectral units
            da = da / ds.dwl
            un = f"{un} {ds.dwl.units}-1"
            # Note: has no impact of toc-relative used

            if dar is not None:
                dar = dar / dsref.dwl

        if toc_relative:
            da = da / da.isel(z=-1)
            ln = f"{ln} (ToC-relative)"
            un = ""

            if dar is not None:
                dar = dar / dar.isel(z=-1)

        lnr = ln
        unr = un
        if ref is not None:
            da = da - dar
            ln = f"{ref_sym} {ln}"
            if ref_relative:
                da = da / dar
                un = ""

        if (da == 0).all():  # skip plotting the reference to avoid the bold teal color
            if ref_plot:
                imr = dar.plot(ax=ax, x=da.dims[1], add_colorbar=False)
            else:
                da.plot(ax=ax, x=da.dims[1], add_colorbar=False).remove()
                ax.set_facecolor("0.9")
        else:
            ims.append(da.plot(ax=ax, x=da.dims[1], add_colorbar=False, norm=norm))

        # Label
        ax.text(
            0.022,
            0.975,
            label,
            va="top",
            ha="left",
            transform=ax.transAxes,
            bbox={"facecolor": "0.7", "alpha": 0.6, "pad": 3},
        )

        # Remove ax labels if not outer
        ax.label_outer()

    # Set all clims to be the same (so we can use a single colorbar)
    clims = list(zip(*[im.get_clim() for im in ims]))
    vmin_min = min(clims[0])
    vmax_max = max(clims[1])
    for im in ims:
        im.set_clim(vmin_min, vmax_max)

    # Add colorbar for reference if plotted
    if imr is not None:
        cbr = fig.colorbar(imr, ax=axs, use_gridspec=True, pad=0.02, aspect=15 * nrows)
        cbr.set_label(f"{lnr} [{_cf_units_to_tex(unr)}]" if unr else lnr)

    # Add single colorbar
    cb = fig.colorbar(im, ax=axs, use_gridspec=True, pad=0.015, aspect=15 * nrows)
    cb.set_label(f"{ln} [{_cf_units_to_tex(un)}]" if un else ln)

    # Note reference in upper left
    if ref is not None:
        axs.flat[0].set_title(f"Reference: {ref_label}", loc="left", fontsize=10)

    # Remove any extras axes
    for ax in axs.flat[len(dsets) :]:
        ax.remove()
        # TODO: add xlabel to row above in this column

    fig.set_tight_layout(False)  # disable since used gridspec


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
