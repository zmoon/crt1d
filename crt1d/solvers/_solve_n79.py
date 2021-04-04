# fmt: off
import numpy as np

from .common import tau_b_fn
from .common import tau_df_fn


short_name = "N79"
long_name = "Norman (1979)"


def solve_n79(
    *,
    psi,
    I_dr0_all, I_df0_all,
    lai,
    leaf_t, leaf_r,
    soil_r,
    K_b_fn,
):

    # tb (tau_b) - exponential transmittance of direct beam radiation through each layer
    dlai = lai[:-1] - lai[1:]  # lai[0] is total LAI, lai[-1] is 0 (toc)
    tb = tau_b_fn(K_b_fn, psi, dlai)

    # td (tau_d) - exponential transmittance of diffuse radiation through each layer
    td = tau_df_fn(K_b_fn, dlai)

    # breakpoint()

    # Preallocate solution arrays
    nz = lai.size
    nwl = leaf_t.size
    I_dr = np.zeros((nz, nwl))
    I_df_d = np.zeros_like(I_dr)
    I_df_u = np.zeros_like(I_dr)
    F = np.zeros_like(I_dr)

    return {
        "I_dr": I_dr,
        "I_df_d": I_df_d,
        "I_df_u": I_df_u,
        "F": F,
    }
