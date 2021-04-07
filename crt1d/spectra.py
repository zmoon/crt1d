"""
Spectral manipulations.
"""
import warnings

import numpy as np
import xarray as xr
from scipy.constants import c
from scipy.constants import h
from scipy.constants import k as k_B
from scipy.constants import N_A
from scipy.integrate import cumtrapz
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline


BAND_DEFNS_UM = {
    "PAR": (0.4, 0.7),
    "NIR": (0.7, 2.5),
    "UV": (0.01, 0.4),
    "solar": (0.3, 5.0),
}


def e_wl_umol(wl_um):
    """J/(μmol photons) at wavelength `wl_um`."""
    # convert wavelength to SI units (for compatibility with the constants)
    wl = wl_um * 1e-6  # um -> m

    e_wl_1 = h * c / wl  # J (per one photon)
    e_wl_mol = e_wl_1 * N_A  # J/mol
    e_wl_umol = e_wl_mol * 1e-6  # J/mol -> J/umol

    return e_wl_umol


def l_wl_planck(T_K, wl_um):
    """Planck radiance.

    Parameters
    ----------
    T_K : float, ndarray
        Temperature (K).
    wl_um : float, ndarray
        Wavelength (μm).
    """
    wl = wl_um * 1e-6  # um -> m
    return (2 * h * c ** 2) / (wl ** 5 * (np.exp(h * c / (wl * k_B * T_K)) - 1))


def l_wl_planck_integ(T_K, wla_um, wlb_um):
    """Numerical integral of Planck radiance from `wla_um` to `wlb_um`.

    Parameters
    ----------
    T_K : float, ndarray
        Temperature (K).
    wla_um : float
        Integration lower bound.
    wlb_um : float
        Upper bound.
    """
    return quad(lambda wl_um: l_wl_planck(T_K, wl_um), wla_um, wlb_um)[0]


def _x_frac_in_bounds(xe, bounds):
    """Fraction in region defined by `bounds` for the bins defined by edges `xe`.
    The calculation includes fractional contributions from bands that are partially
    within the `bounds`.

    For bins of irradiance (W m-2, not W m-2 μm-1),
    this can be used as weights to integrate over a region.

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


def avg_optical_prop(y, bounds, *, x=None, xe=None, light="planck", **light_kwargs):
    r"""Average reflectance or transmittance over some region,
    from a spectrum or binned (smeared) spectrum.

    The average value depends on the spectrum of the illuminating light
    as well as the property's spectrum.

    Parameters
    ----------
    y : array_like
        Reflectance or transmittance spectrum.
    bounds : array_like
        Region lower and upper bounds.
    x : array_like, optional
        Wavelength values where `y` is defined.
        (Bin centers, or coordinates of original spectrum.)
        If `x` is provided (instead of `xe`), `y`\(`x`) is first smeared with :func:`smear_tuv`
        into 200 equally spaced bins over the `bounds` region.
    xe : array_like, optional
        Bin edges.
        Don't provide `x` if using `xe`.
    light : str, array_like, callable
        Method for constructing the light weights used in the weighted average.

        Default method integrates 6000 K Planck radiance over each bin
        using :func:`l_wl_planck_integ`.

        Can be an array-like of weights (e.g., irradiances).

        Can be function that can take `x` as its first parameter to provide
        a weight for `y`\(`x`).
    **light_kwargs
        For the `light` method.
        Currently this only includes ``T_K`` for ``light='planck'``.
    """
    y = np.asarray(y)
    if x is not None and xe is not None:
        raise ValueError("only provide one of `x` and `xe`.")
    if x is None and xe is None:
        raise ValueError("`x` or `xe` must be provided!")

    # Spectrum -- `y` defined at `x`
    if x is not None:
        x = np.asarray(x)
        assert x.size == y.size
        # Now bin (temporary!? until working weights into smear)
        # TODO: arg to control nbins and/or dx resolution (1 nm?)?
        xe = np.linspace(*bounds, 201)
        y = smear_tuv(x, y, xe)

    # Already been binned -- use the `xe` edges
    if xe is not None:
        xe = np.asarray(xe)
        assert xe.size == y.size + 1
        x = (xe[:-1] + xe[1:]) / 2  # midpts

    w = _x_frac_in_bounds(xe, bounds)  # initial weights

    # Add weights based on light spectrum
    if isinstance(light, str):
        if light == "planck":
            T_K = light_kwargs.get("T_K", 6000)
            w *= [l_wl_planck_integ(T_K, a, b) for a, b in zip(xe[:-1], xe[1:])]
        elif light == "uniform":
            w *= np.ones_like(y)
        else:
            raise ValueError("invalid choice of `light`")
    elif callable(light):
        w *= light(x, **light_kwargs)
    else:  # assume array-like
        w *= np.asarray(light)

    return (y * w).sum() / w.sum()  # weighted average


def _smear_tuv_1(x, y, bin):
    r"""Smear `y` values at positions `x` over the region defined by `bin`.
    Returns a single value, corresponding to the (trapezoidally) integrated average of
    :math:`y(x)` in the bin.

    Parameters
    ----------
    x : array_like
        the original grid
    y : array_like
        the values :math:`y(x)` to be smeared
    bin : array_like
        a lower and upper bound: (:math:`x_l`, :math:`x_u`)
    """
    xl, xu = bin  # shadowing a builtin but whatever
    area = 0
    for k in range(x.size - 1):  # TODO: could try subsetting first instead of over whole grid
        if x[k + 1] < xl or x[k] > xu:  # outside window
            continue

        a1 = max(x[k], xl)
        a2 = min(x[k + 1], xu)

        slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
        b1 = y[k] + slope * (a1 - x[k])
        b2 = y[k] + slope * (a2 - x[k])
        area = area + (a2 - a1) * (b2 + b1) / 2

    return area / (xu - xl)


def smear_tuv(x, y, bins):
    r"""Smear `y`\(`x`) into `bins`,
    using the TUV method.
    Returns an array of in-bin *y*-values ``ynew``, such that
    :math:`\sum_i y_{\text{new}, i} \Delta x_i` (``(ynew * np.diff(dx)).sum()``)
    is equal to the original trapezoidal integral of *y(x)*
    over the range [``bins[0]``, ``bins[-1]``], ie.,

    Each value in the result is the (trapezoidally) integrated average of :math:`y(x)`
    in the corresponding bin (:math:`x_{l,i}`, :math:`x_{u,i}`).

    .. math::
       \frac{\int_{x_l}^{x_u} y(x) \, dx}{x_u - x_l}

    Parameters
    ----------
    x : array_like
        Coordinates of the `y` values (the original grid).
    y : array_like
        Values :math:`y(x)` to be smeared/binned.
    bins : array_like
        Bin edges.

    Returns
    -------
    ynew : array_like
        New values, valid for each bin (size ``bins.size - 1``).

    Notes
    -----
    Implementation based on
    `F0AM's implementation <https://github.com/AirChem/F0AM/blob/281f80adbdaeaf580d4757f8bf6ca02407251974/Chem/Photolysis/IntegrateJ.m#L193-L217>`_
    of TUV's un-named algorithm.
    It works by applying cumulative trapezoidal integration to the original data,
    interpolating within `x` so that :math:`x_l` and :math:`x_u` don't have to be on the original `x` grid.
    """
    ynew = np.zeros(bins.size - 1)
    # TODO: more efficient looping, maybe `while` over `x`
    for i, bin_ in enumerate(zip(bins[:-1], bins[1:])):
        ynew[i] = _smear_tuv_1(x, y, bin_)
    return ynew  # valid for band, not at left edge


def smear_trapz_interp(x, y, bins, *, k=3, interp="F"):
    r"""Smear `y`\(`x`) into `bins`,
    using
    `spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html>`_
    interpolation and trapezoidal integration.

    Parameters
    ----------
    x : array_like
        Coordinates of the `y` values (the original grid).
    y : array_like
        Values :math:`y(x)` to be smeared/binned.
    bins : array_like
        Bin edges.
    k : int
        Degree of the spline used (1--5). The spline passes through all data points.
    interp : str
        ``'F'`` to interpolate the cumulative trapz integral,
        or ``'f'`` to interpolate the `y` data.

    Returns
    -------
    ynew : array_like
        New values, valid for each bin (size ``bins.size - 1``).

    """
    if interp == "f":
        spl = InterpolatedUnivariateSpline(x, y, k=k)
        y_bins = spl(bins)
        Ynew = cumtrapz(y_bins, x=bins, initial=0)
        dYnew = np.diff(Ynew)

    elif interp == "F":
        Y0 = cumtrapz(y, x, initial=0)
        spl = InterpolatedUnivariateSpline(x, Y0, k=k)
        dYnew = spl(bins[1:]) - spl(bins[:-1])

    else:
        raise ValueError(f"invalid `interp` {interp!r}")

    dxnew = np.diff(bins)

    return dYnew / dxnew


def smear_avg_optical_prop(x, y, bins, **kwargs):
    r"""Smear `y`\(`x`) into `bins`,
    using :func:`avg_optical_prop` with `x` as ``x``.
    `**kwargs` are passed on to :func:`avg_optical_prop`.

    Parameters
    ----------
    x : array_like
        Coordinates of the `y` values (the original grid).
    y : array_like
        Values :math:`y(x)` to be smeared/binned.
    bins : array_like
        Bin edges.

    Returns
    -------
    ynew : array_like
        New values, valid for each bin (size ``bins.size - 1``).
    """
    ynew = np.zeros((bins.size - 1))
    for i, bounds in enumerate(zip(bins[:-1], bins[1:])):
        # TODO: allow `xe` option somehow?
        ynew[i] = avg_optical_prop(y, bounds, x=x, **kwargs)

    return ynew


def _smear_arr(x, y, bins, *, method, **method_kwargs):
    if method == "tuv":
        ynew = smear_tuv(x, y, bins)

    elif method == "trapz_interp":
        ynew = smear_trapz_interp(x, y, bins, **method_kwargs)

    elif method == "avg_optical_prop":
        ynew = smear_avg_optical_prop(x, y, bins, **method_kwargs)

    else:
        raise ValueError(f"invalid `method` {method!r}")

    return ynew


def _smear_da_dv(da, bins, *, xname, xname_out, method, **method_kwargs):
    """Returns an :class:`xarray.Dataset` ``data_vars`` tuple."""
    x = da[xname].values
    y = da.values

    ynew = _smear_arr(x, y, bins, method=method, **method_kwargs)

    return (xname_out, ynew, da.attrs)


def _smear_ds(ds, bins, *, xname="wl", xname_out=None, method="tuv", **method_kwargs):
    """Smear spectra (with coordinate variable `xname`) to new `bins`.

    The returned :class:`xarray.Dataset` consists of the data variables who have the coordinate
    variable with ``name`` `xname`.

    .. warning::
       Smearing irradiance (W m-2) will not have the desired result.
       Instead, be sure to smear *spectral* irradiance (W m-2 μm-1),
       e.g., using :func:`smear_si`,
       to preserve the spectrum shape and total integrated irradiance.

    Parameters
    ----------
    ds : xr.Dataset
    bins : array_like
        Bin edges for the smeared spectrum.
    xname_out : str, optional
        By default, same as `xname`.
    method : str
        Smear method.
        ``'tuv'`` for :func:`smear_tuv` or ``'trapz_interp'`` for :func:`smear_trapz_interp`.
    **method_kwargs
        Passed on to the smear method function.
    """
    bins = np.asarray(bins)
    da_x = ds[xname]
    dx = np.diff(bins)
    xnewc = bins[:-1] + 0.5 * dx

    if xname_out is None:
        xname_out = xname

    new_data_vars = {
        vn: _smear_da_dv(
            ds[vn], bins, xname=xname, xname_out=xname_out, method=method, **method_kwargs
        )
        for vn in ds.variables
        if xname in ds[vn].coords and vn not in ds.coords
    }

    da_xo = ds[xname_out]
    xe_name = f"{da_xo.name}e"
    xe_dims = [f"{d}e" for d in da_xo.dims]
    xe_attrs = {
        "long_name": f"{da_x.attrs['long_name']} band edges",
        "units": da_xo.attrs["units"],
    }
    return xr.Dataset(
        coords={
            da_xo.name: (da_xo.dims, xnewc, da_xo.attrs),
            xe_name: (xe_dims, bins, xe_attrs),
        },
        data_vars=new_data_vars,
    )


def smear(y, bins, *, x="wl", xname_out="wl", method="tuv", **method_kwargs):
    """Smear `y` into `bins`.

    This function dispatches to the different smearing method functions,
    which take array-like inputs and can be used directly instead if desired.

    If `y` is array-like, this returns a :class:`numpy.ndarray`.

    If `y` is an xarray type, this returns an :class:`xarray.Dataset`.

    Parameters
    ----------
    y : array_like, xr.DataArray, xr.Dataset
        Variable(s) to smear.
        If :class:`xarray.Dataset`, all variables that have coordinate `x` will be included.
    bins : array_like
        Bin edges for the smeared spectrum.
    x : str, array_like, xr.DataArray
        Coordinates at which the `y` values are valid/defined.
        If `y` is `array`, `x` must be provided as an `array` of the same size.
        If `y` is an xarray type, providing the ``name`` as a string is sufficient.
    xname_out : str
        Used to label the new dimension of `y` is an `xarray.Dataset`, for example.
    method : str
        Smear method.

        * ``'tuv'``: :func:`smear_tuv`

        * ``'trapz_interp'``: :func:`smear_trapz_interp`

        * ``'avg_optical_prop'``: :func:`smear_avg_optical_prop`
    **method_kwargs
        Passed on to the smear method function.
    """
    # Convert DataArray to Dataset if necessary
    if isinstance(y, xr.DataArray):
        y = xr.Dataset(data_vars={y.name: y})

    if isinstance(y, xr.Dataset):
        assert isinstance(x, str), "specify `x` as string"
        # TODO: support `x` as DataArray?
        return _smear_ds(y, bins, xname=x, xname_out=xname_out, method=method, **method_kwargs)

    else:  # assume array-like
        assert not isinstance(x, str), "`x` array must be provided if `y` is array"
        y = np.asarray(y)
        x = np.asarray(x)
        bins = np.asarray(bins)
        assert x.size == y.size, "`x` and `y` must be same size"
        return _smear_arr(x, y, bins, method=method, **method_kwargs)


def smear_si(ds, bins, *, xname_out="wl", **kwargs):
    """Helper function to smear spectral irradiance
    and then compute in-bin irradiance in the new bins.
    `**kwargs` can include ``method`` and ``**method_kwargs`` (see :func:`smear`).

    .. note::
       `ds` must have variables ``'SI_dr'`` (direct spectral irradiance)
       and ``'SI_df'`` (diffuse).

    Parameters
    ----------
    ds : xr.Dataset
    """
    # coordinate variable for the spectral irradiances
    xname = list(ds["SI_dr"].coords)[0]

    # smear the spectral irradiances
    ds_new = _smear_ds(
        ds[["SI_dr", "SI_df", xname_out]], bins=bins, xname=xname, xname_out=xname_out, **kwargs
    )

    # compute irradiances
    dwl = np.diff(ds_new.wle)
    units = "W m-2"
    ds_new["I_dr"] = (
        xname_out,
        ds_new.SI_dr * dwl,
        {"units": units, "long_name": "Direct irradiance"},
    )
    ds_new["I_df"] = (
        xname_out,
        ds_new.SI_df * dwl,
        {"units": units, "long_name": "Diffuse irradiance"},
    )
    ds_new["dwl"] = (xname_out, dwl, {"units": "μm", "long_name": "Wavelength band width"})
    # TODO: ensure attrs consistency somehow

    return ds_new


def _edges_from_centers(x):
    """Estimate locations of the bin edges by extrapolating the grid spacing at the edges.

    .. warning::
       Note that the true bin edges could be different---this is just a guess!

    From specutils:
    https://github.com/astropy/specutils/blob/9ce88d6be700a06e888d9d0cdd04c0afc2ae85f8/specutils/spectra/spectral_axis.py#L46-L55
    """
    extend_left = np.r_[2 * x[0] - x[1], x]
    extend_right = np.r_[x, 2 * x[-1] - x[-2]]
    edges = (extend_left + extend_right) / 2

    return edges


def _interpret_dx_relative_spectrum(ydx, x, *, midpt=True):
    r"""For spectrally distributed values `ydx`,
    equivalent to :math:`y(x)/dx` (e.g., spectral irradiance in W m-2 μm-1),
    determine corresponding *y* values.

    .. note::
       No need to use this if the bin edges are known!
       (Just multiply by the bin edges :math:`\delta x`.)

    Returns *y* (i.e., *ydx dx*), and corresponding *x* values and bin widths (*dx*).

    With the midpoint method (``midpt=True``), these will have have size *n*-1 wrt. `ydx`.
    Summing these *y* values is equivalent to the midpoint Riemann sum of the original
    *y(x)/dx*.

    With the edges-from-centers method (``midpt=False``; uses :func:`_edges_from_centers`),
    the return arrays will be size *n*.
    Estimating the edges allows the original *y(x)/dx* values to be used to compute *y(x*).

    Returns
    -------
    tuple(array_like)
        ``y``, ``x_y``, ``dx``
    """
    if midpt:
        xe = x
        dx = np.diff(x)

        x_y = xe[:-1] + 0.5 * dx  # x positions of y values (integrated y/dx)
        y = (ydx[:-1] + ydx[1:]) / 2 * dx

    else:  # edges-from-centers method
        xe = _edges_from_centers(x)  # n+1 values
        dx = np.diff(xe)

        x_y = x  # original grid
        y = ydx * dx

    return y, x_y, dx  # TODO: return `xe` as well, maybe as named tuple?


def plot_binned(x, y, xc, yc, dx, *, ax=None, xtight="orig"):
    """Compare an original spectrum to binned one.

    If `x` and `y` are :class:`xarray.DataArray`, their attrs will be used to label the plot.

    Parameters
    ----------
    x, y : array_like
        original/actual values at wavelengths (original spectra)
    xc, yc : array_like
        values at wave band centers
    dx : array_like
        wave band widths, same size as `xc`, `yc`.
    """
    import matplotlib.pyplot as plt
    from .utils import cf_units_to_tex

    spectrum_ls = "r-" if x.size > 150 else "r.-"
    if xtight == "orig":
        xbounds = (x[0], x[-1])
    elif xtight == "bins":
        xbounds = (xc[0] - 0.5 * dx[0], xc[-1] + 0.5 * dx[-1])
    else:
        raise ValueError("invalid `xtight`")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    ax.bar(
        xc,
        yc,
        width=dx,
        label=f"binned\n$N = {len(xc)}$",
        align="center",
        alpha=0.7,
        edgecolor="blue",
    )

    ax.plot(x, y, spectrum_ls, lw=1.5, label="spectrum")

    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        unx = cf_units_to_tex(x.attrs["units"])
        lnx = x.attrs["long_name"]
        xlabel = f"{lnx} [{unx}]"
        uny = cf_units_to_tex(y.attrs["units"])
        lny = y.attrs["long_name"]
        ylabel = f"{lny} [{uny}]"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set(xlabel="$x$", ylabel="$y$")

    ax.set_xlim(xbounds)
    ax.legend()
    fig.tight_layout()


def plot_binned_ds(ds0, ds, *, yname=None, **kwargs):
    """Compare an original spectrum in `ds0` to binned one in `ds`.
    Uses :func:`plot_binned` with auto-detection of ``x``, ``dx``, and ``xc``
    (though `yname`, the ``name`` of the spectrum to be plotted, must be provided).
    `**kwargs` are passed on to :func:`plot_binned`.
    """
    if yname is None:
        raise ValueError("`yname` must be provided in order to plot the spectrum")

    y = ds0[yname]
    xname = list(y.coords)[0]
    x = ds0[xname]

    yc = ds[yname]
    xcname = list(yc.coords)[0]
    xc = ds[xcname]

    # get dx, either from stored variable or by comnputing from edges
    try:
        dx = ds[f"d{xcname}"]
    except KeyError:
        xe = ds[f"{xcname}e"]
        dx = np.diff(xe)

    plot_binned(x, y, xc, yc, dx, **kwargs)


if __name__ == "__main__":
    pass
