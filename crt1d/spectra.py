"""
Spectral manipulations.
"""
import numpy as np
import xarray as xr
from scipy.constants import c
from scipy.constants import h
from scipy.constants import k as k_B
from scipy.integrate import cumtrapz
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline


def l_wl_planck(T_K, wl_um):
    """Planck radiance.

    Parameters
    ----------
    T_K : float, ndarray
        Temperature (K).
    wl_um : float, ndarray
        Wavelength (μm).
    """
    wl = wl_um * 1e-6  # -> m
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
    return quad(l_wl_planck, wla_um, wlb_um)[0]


def smear_tuv_1(x, y, bin):
    r"""Smear `y` values at positions `x` over the region defined by `bin`.
    Returns a single value, corresponding to the (trapezoidally) integrated average of
    :math:`y(x)` in the bin.

    .. math::
       \frac{\int_{x_l}^{x_u} y(x) \, dx}{x_u - x_l}

    Parameters
    ----------
    x : array_like
        the original grid
    y : array_like
        the values :math:`y(x)` to be smeared
    bin : tuple(float)
        a lower and upper bound: (:math:`x_l`, :math:`x_u`)

    Notes
    -----
    Implementation based on
    `F0AM's implementation <https://github.com/AirChem/F0AM/blob/281f80adbdaeaf580d4757f8bf6ca02407251974/Chem/Photolysis/IntegrateJ.m#L193-L217>`_
    of TUV's un-named algorithm.
    It works by applying cumulative trapezoidal integration to the original data,
    interpolating within `x` so that :math:`x_l` and :math:`x_u` don't have to be on the original `x` grid.
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
    r"""Use :func:`smear_tuv_1` to bin `y` values at `x` positions into `bins` (new *x*-bin edges).
    Returns an array of in-bin *y*-values ``ynew``, such that
    :math:`\sum_i y_{\text{new}, i} \Delta x_i`
    is equal to the original trapezoidal integral of *y(x)* over the range [``bins[0]``, ``bins[-1]``].
    """
    ynew = np.zeros(bins.size - 1)
    for i, bin_ in enumerate(zip(bins[:-1], bins[1:])):
        ynew[i] = smear_tuv_1(x, y, bin_)
    return ynew  # valid for band, not at left edge


def smear_trapz_interp(x, y, bins, *, k=3, interp="F"):
    r"""Smear `y`\(`x`) to the centers of `bins`,
    using
    `spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html>`_
    interpolation and trapezoidal integration.

    Parameters
    ----------
    k : int
        Degree of the spline used (1--5). The spline passes through all data points.
    interp : str
        ``'F'`` to interpolate the cumulative trapz integral, or ``'f'`` to interpolate the `y` data.
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


def _smear_da(da, bins, *, xname, xname_out, method, **method_kwargs):
    """Returns an :class:`xarray.Dataset` ``data_vars`` tuple."""
    x = da[xname].values
    y = da.values

    if method == "tuv":
        ynew = smear_tuv(x, y, bins)

    elif method == "trapz_interp":
        ynew = smear_trapz_interp(x, y, bins, **method_kwargs)

    else:
        raise ValueError(f"invalid `method` {method!r}")

    return (xname_out, ynew, da.attrs)


def smear_ds(ds, bins, *, xname="wl", xname_out=None, method="tuv", **method_kwargs):
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
    da_x = ds[xname]
    dx = np.diff(bins)
    xnewc = bins[:-1] + 0.5 * dx

    if xname_out is None:
        xname_out = xname

    new_data_vars = {
        vn: _smear_da(
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


def smear_si(ds, bins, *, xname_out="wl", **kwargs):
    """Smear spectral irradiance, computing in-bin irradiance in the new bins.
    `**kwargs` are passed to :func:`smear_ds`.
    `ds` must have variables ``'SI_dr'`` (direct spectral irradiance), ``'SI_df'`` (diffuse).
    """
    # coordinate variable for the spectral irradiances
    xname = list(ds["SI_dr"].coords)[0]

    # smear the spectral irradiances
    ds_new = smear_ds(
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


def edges_from_centers(x):
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


def interpret_spectrum(ydx, x, *, midpt=True):
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

    With the edges-from-centers method (``midpt=False``; uses :func:`edges_from_centers`),
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
        y = (ydx[:-1] + y[1:]) / 2 * dx

    else:  # edges-from-centers method
        xe = edges_from_centers(x)  # n+1 values
        dx = np.diff(xe)

        x_y = x  # original grid
        y = ydx * dx

    return y, x_y, dx


def plot_binned(x, y, xc, yc, dx):
    """Compare an original spectrum to binned one.

    If `x` and `y` are :class:`xarray.DataArray`, their attrs will be used to label the plot.

    Parameters
    ----------
    x, y :
        original/actual values at wavelengths (original spectra)
    xc, yc :
        values at wave band centers
    dx :
        wave band widths
    """
    import matplotlib.pyplot as plt
    from .utils import cf_units_to_tex

    spectrum_ls = "r-" if x.size > 150 else "r.-"
    xbounds = (x[0], x[-1])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(xc, yc, width=dx, label="binned", align="center", alpha=0.7, edgecolor="blue")

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

    ax.set_xlim(xbounds)

    ax.legend()

    fig.tight_layout()


def plot_binned_ds(ds0, ds, *, yname=None):
    """Compare an original spectrum in `ds0` to binned one in `ds`.
    Uses :func:`plot_binned` with auto-detection of ``x``, ``dx``, and ``xc``
    (though `yname`, the ``name`` of the spectrum to be plotted, must be provided).
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

    plot_binned(x, y, xc, yc, dx)


if __name__ == "__main__":
    pass
