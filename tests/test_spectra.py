"""
Test crt1d.spectra
"""

import math

import numpy as np
import pytest

import crt1d as crt


@pytest.mark.parametrize("T_K", (1000, 5000, 5800, 6000))
def test_l_wl_planck_integ_T(T_K):
    from scipy.constants import sigma

    # Estimate total irradiance
    with np.errstate(over="ignore"):  # ignore overflow in `np.exp` warning
        est = math.pi * 1e-6 * crt.spectra.l_wl_planck_integ(T_K, 0, math.inf)

    # Compare to theoretical
    assert math.isclose(est, sigma * T_K**4)


@pytest.mark.parametrize(
    "xe,bounds,expected",
    [
        pytest.param(np.r_[0, 1, 2, 3], (0, 3), [1, 1, 1], id="all in"),
        pytest.param(np.r_[0, 1, 2, 3], (0.5, 2.2), [0.5, 1, 0.2], id="fractions"),
        pytest.param(np.r_[0, 1, 2, 3], (0.5, 2.0), [0.5, 1, 0], id="one out (edge in)"),
    ],
)
def test_x_frac_in_bounds(xe, bounds, expected):
    actual = crt.spectra._x_frac_in_bounds(xe, bounds)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "x,bins,expected",
    [
        pytest.param(np.r_[0, 1, 2, 3, 4], (0, 2, 4), [1, 3], id="equal bin edges in x"),
        pytest.param(np.r_[0, 1, 2, 3, 4], (0, 1, 4), [0.5, 2.5], id="unequal bin edges in x"),
        pytest.param(np.r_[0, 1, 2, 3, 4], (0, 1, 6), [0.5, 7.5 / (6 - 1)], id="bin edge beyond x"),
        pytest.param(np.r_[0, 1], (2, 3), [0], id="single bin fully outside"),
    ],
)
def test_smear_tuv(x, bins, expected):
    # We use the function y = x as the original spectrum to be smeared
    y = x
    actual = crt.spectra.smear_tuv(x, y, bins)
    np.testing.assert_allclose(actual, expected)
    actual2 = crt.spectra.smear_tuv2(x, y, bins)
    np.testing.assert_allclose(actual2, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        pytest.param(np.r_[0.5, 1.5, 2.5], [0, 1, 2, 3], id="equal spacing"),
        pytest.param(np.r_[0.5, 1.5, 2], [0, 1, 1.75, 2.25], id="variable spacing"),
    ],
)
def test_edges_from_centers(x, expected):
    actual = crt.spectra._edges_from_centers(x)
    np.testing.assert_allclose(actual, expected)

    dx = np.diff(x)
    xcnew = (actual[:-1] + actual[1:]) / 2  # new centers
    if not all(dx == dx[0]):  # variable grid
        assert not np.all(x == xcnew)
    else:
        np.testing.assert_allclose(x, xcnew)


def test_e_wl_umol():
    from scipy.constants import N_A

    N_umol = N_A / 1e6
    assert math.isclose(crt.spectra.e_wl_umol(3), 6.621e-20 * N_umol, rel_tol=1e-4)
