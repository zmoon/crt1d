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
    assert math.isclose(est, sigma * T_K ** 4)


@pytest.mark.parametrize(
    "xe,bounds,expected",
    [
        pytest.param(np.r_[0, 1, 2, 3], (0, 3), np.r_[1, 1, 1], id="all in"),
        pytest.param(np.r_[0, 1, 2, 3], (0.5, 2.2), np.r_[0.5, 1, 0.2], id="fractions"),
        pytest.param(np.r_[0, 1, 2, 3], (0.5, 2.0), np.r_[0.5, 1, 0], id="one out (edge in)"),
    ],
)
def test_x_frac_in_bounds(xe, bounds, expected):
    np.testing.assert_allclose(crt.spectra._x_frac_in_bounds(xe, bounds), expected)
