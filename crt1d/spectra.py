"""
Spectral manipulations.
"""
import numpy as np


def smear_tuv_1(x, y, xgl, xgu):
    """Smear `y`(`x`) over the region defined by [`xgl`, `xgu`].

    Parameters
    ----------
    x : array_like
        the original grid
    y : array_like
        the values y(x) to be smeared
    xgl : float
        left side x bound
    xgu : float
        right side x bound

    Notes
    -----
    Implementation based on F0AM's implementation of TUV's un-named algorithm.
    It works by applying cumulative trapezoidal integration to the original data,
    but extrapolating so that xgl and xgu don't have to be on the original x grid.
    """
    area = 0
    for k in range(x.size - 1):  # TODO: could try subsetting first instead of over whole grid
        if x[k + 1] < xgl or x[k] > xgu:  # outside window
            continue

        a1 = max(x[k], xgl)
        a2 = min(x[k + 1], xgu)

        slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
        b1 = y[k] + slope * (a1 - x[k])
        b2 = y[k] + slope * (a2 - x[k])
        area = area + (a2 - a1) * (b2 + b1) / 2

    yg = area / (xgu - xgl)

    return yg


def smear_tuv(x, y, bins):
    """Use :func:`smear_tuv_1` to bin `y`(`x`) into `bins` (new x-bin edges).
    Returns an array of in-bin y-values ynew, such that the sum of ynew*dwl
    is equal to the original integral of y(x).
    """
    ynew = np.zeros(bins.size - 1)
    for i, (xgl, xgu) in enumerate(zip(bins[:-1], bins[1:])):
        ynew[i] = smear_tuv_1(x, y, xgl, xgu)
    return ynew  # valid for band, not at left edge


def smear_trapz_interp():
    raise NotImplementedError


def smear_ds(ds, *, method="tuv"):
    """"""
    raise NotImplementedError


def edges_from_centers(x):
    """Estimate locations of the bin edges by extrapolating the grid spacing on the edges.
    Note that the true bin edges could be different--this is just a guess!

    From specutils:
    https://github.com/astropy/specutils/blob/9ce88d6be700a06e888d9d0cdd04c0afc2ae85f8/specutils/spectra/spectral_axis.py#L46-L55
    """
    extend_left = np.r_[2 * x[0] - x[1], x]
    extend_right = np.r_[x, 2 * x[-1] - x[-2]]
    edges = (extend_left + extend_right) / 2

    return edges


if __name__ == "__main__":
    pass
