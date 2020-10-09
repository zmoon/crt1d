"""
Common functions used by canopy RT solvers.
"""
import numpy as np
import scipy.integrate as si


def tau_b_fn(K_b_fn, psi, lai_val):
    """Transmittance of direct beam through the medium (foliage elements)

    We need to be able to integrate over different zenith angles (to calulate tau_df),
    so we supply K_b_fn instead of K_b

    Parameters
    ----------
    K_b_fn : function
        := G_fn(psi)/cos(psi) where G_fn is "big G" for the chosen leaf angle distribution function.
    psi : float or array_like
        Solar zenith angle (radians)
    lai_val : float
        a value of LAI from the cumulative LAI profile (~ an extinction optical thickness/depth)
    """
    return np.exp(-K_b_fn(psi) * lai_val)


def tau_df_fn(K_b_fn, lai_val):
    """Transmissivity for diffuse light through the medium

    Weighted hemispherical integral of direct beam transmissivity tau_b.
    Isotropy assumption implicit.
    Note that it does not depend on psi.

    ref. Campbell & Norman eq. 15.5
    """
    tau_b = lambda psi_, L_: tau_b_fn(K_b_fn, psi_, L_)
    f = lambda psi_: tau_b(psi_, lai_val) * np.sin(psi_) * np.cos(psi_)
    return 2 * si.quad(f, 0, np.pi / 2, epsrel=1e-9)[0]


# TODO: vectorize tau_df_fn


def K_df_fn(K_b_fn, lai_tot):
    """K_d from K_b(psi) and total LAI, using `tau_df_fn`."""
    tau_df = tau_df_fn(K_b_fn, lai_tot)
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
