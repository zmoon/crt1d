"""
Common functions used by canopy RT solvers.
"""
import math

import numpy as np
import scipy.integrate as integrate


def tau_b_fn(K_b_fn, psi, lai):
    r"""Transmittance of direct beam through foliage layers(s) with LAI `lai`.

    We need to be able to integrate over different zenith angles (to calulate :math:`\tau_d`),
    so we supply `K_b_fn` instead of ``K_b`` (:math:`K_b`) itself.

    Parameters
    ----------
    K_b_fn : function
        := ``G_fn(psi)/cos(psi)`` where ``G_fn`` computes :math:`G`
        for the chosen leaf angle distribution function based on :math:`\psi`.
    psi : float or array_like
        Solar zenith angle (radians).
    lai : float or array_like
        LAI.
    """
    return np.exp(-K_b_fn(psi) * lai)


def _tau_df_fn_scalar(K_b_fn, lai_val):
    r""":math:`\tau_d` for scalar LAI value."""
    from functools import partial

    tau_b_psi = partial(tau_b_fn, K_b_fn=K_b_fn, lai=lai_val)

    f = lambda psi: tau_b_psi(psi=psi) * np.sin(psi) * np.cos(psi)  # noqa: E731
    return 2 * integrate.quad(f, 0, np.pi / 2, epsrel=1e-9)[0]


def _tau_df_fn_scalar_9sky(K_b_fn, lai_val):
    r""":math:`\tau_d` for scalar LAI value using 9 angles only (common in land models)."""
    from functools import partial

    tau_b_psi = partial(tau_b_fn, K_b_fn=K_b_fn, lai=lai_val)

    tau_d = 0
    for sza in [5, 15, 25, 35, 45, 55, 65, 75, 85]:
        psi = math.radians(sza)
        tau_d += tau_b_psi(psi=psi) * math.sin(psi) * math.cos(psi)

    tau_d *= 2 * math.radians(10)  # 90 -> 180; multiplication by dpsi

    return tau_d


def tau_df_fn(K_b_fn, lai, *, method="quad"):
    r"""Transmittance of diffuse light through foliage layer(s) with LAI `lai`.

    Weighted hemispherical integral of direct beam transmissivity :math:`\tau_b`.
    Isotropy assumption implicit.
    Note that it does not depend on :math:`\psi`.

    Parameters
    ----------
    lai : float or array_like
        LAI, one or multiple values.
    method : {'quad', '9sky'}

    References
    ----------
    * Campbell & Norman eq. 15.5 :cite:`campbell_introduction_2012`
    """
    if method == "quad":
        f = _tau_df_fn_scalar
    elif method == "9sky":
        f = _tau_df_fn_scalar_9sky
    else:
        raise ValueError("invalid `method`. Valid options are 'quad' and '9sky'.")

    if np.isscalar(lai):
        res = f(K_b_fn, lai)
    else:
        res = np.zeros_like(lai)
        for i, lai_val in enumerate(lai):
            res[i] = f(K_b_fn, lai_val)

    return res


def K_df_fn(K_b_fn, lai_tot, **kwargs):
    r""":math:`K_d` from :math:`K_b(\psi)` and total LAI, using :func:`tau_df_fn`.
    `**kwargs` passed on to :func:`tau_df_fn`.
    """
    tau_df = tau_df_fn(K_b_fn, lai_tot, **kwargs)
    return -np.log(tau_df) / lai_tot


# TODO: mu version of tau_df and tau_b (or mu/psi choice as input)
# should also do for G
# and G integral fn (like in Gu-Barr)

# class Solver():
#     """Class to call specified solver with necessary arguments
#     and loop over wavelengths??.

#     probably leave them with loops in the solvers for now,
#     since some have scheme-specific params that can stay outside loop to save time

#     but that is really only B-L so maybe should go ahead...
#     can do some special stuff for B-L
#     zq does have some too (tau profile)

#     """

#     def __init__(self, scheme_ID,
#         cnpy_rad_state, cnpy_descrip):

#         pass

#     def solve(self, cnpy_rad_state):

#         #> allocate arrays in which to save the solutions for each band
#         # I_dr_all = np.zeros((lai.size, wl.size))
#         # I_df_d_all = np.zeros_like(I_dr_all)
#         # I_df_u_all = np.zeros_like(I_dr_all)
#         # F_all = np.zeros_like(I_dr_all)
#         s = (lai.size, wl.size)  # < make pylint shut up until it supports _like()
#         I_dr_all   = np.zeros(s)
#         I_df_d_all = np.zeros(s)
#         I_df_u_all = np.zeros(s)
#         F_all      = np.zeros(s)
