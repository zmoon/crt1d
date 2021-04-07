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
