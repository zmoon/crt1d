# fmt: off
import numpy as np

from .common import tau_b_fn
from .common import tau_df_fn


short_name = "N79"
long_name = "Norman (1979)"


def tdma(a, b, c, d):
    """Tridiagonal matrix algo.
    for `a` below diagonal, `b` diagonal, `c` above diagonal and `d` RHS.

    Parameters
    ----------
    a, b, c, d : array
        Same size!

    Reference: https://github.com/gbonan/bonanmodeling/blob/master/sp_14_03/tridiagonal_solver.m
    """
    # TODO: something is not right! soln not same as solving with scipy
    n = a.size

    # Forward sweep 1
    e = np.zeros_like(a)
    e[0] = c[0] / b[0]
    for i in range(1, n-1):
        e[i] = c[i] / (b[i] - a[i] * e[i-1])

    # Forward sweep 2
    f = np.zeros_like(a)
    f[0] = d[0] / b[0]
    for i in range(1, n-1):
        f[i] = (d[i] - a[i] * f[i-1]) / (b[i] - a[i] * e[i-1])

    # Backwards substitution for solution
    u = np.zeros_like(a)
    u[-1] = f[-1]
    for i in range(n-2, -1, -1):
        u[i] = f[i] - e[i] * u[i+1]

    return u


def solve_n79(
    *,
    psi,
    I_dr0_all, I_df0_all,
    lai,
    leaf_t, leaf_r,
    soil_r,
    K_b_fn,
):
    """Norman (1979) canopy radiation solution.

    References
    ----------
    * :cite:`bonan_climate_2019`
      `SP 14.3 <https://github.com/gbonan/bonanmodeling/tree/master/sp_14_03>`_
    """

    K_b = K_b_fn(psi)

    # New variable names for consistency with Bonan
    albsoib = soil_r  # soil albedo for direct
    albsoid = soil_r  # soil albedo for diffuse (assume same like in the Bonan SP)
    swskyb = I_dr0_all
    swskyd = I_df0_all
    rho = leaf_r
    tau = leaf_t

    # My setup is a bit different from Bonan's because he has cLAI nonzero at top layer
    # since he has dlai defined on the interface levels and then cLAI within.
    dlai = lai[:-1] - lai[1:]  # need 2nz for equation arrays
    # dlai = np.r_[lai[:-1] - lai[1:], 0]  # need 2(nz+1)

    # tb (tau_b) - exponential transmittance of direct beam radiation through each layer
    # Note: `lai[0]` is total LAI, `lai[-1]` is 0 (toc)
    tb = tau_b_fn(K_b_fn, psi, dlai)
    tbcum = np.exp(-K_b * lai)
    # For Bonan, `tbcum` starts at `iv=canopy.nbot-1` (`== canopy.nsoi`, top soil level).
    # https://github.com/gbonan/bonanmodeling/blob/a10cf764013be58c2def1dbe7c7e52a3213e061e/sp_14_03/sp_14_03.m#L210
    # Thus it has one more value than `tb`.
    assert tb.size == tbcum.size - 1

    # td (tau_d) - exponential transmittance of diffuse radiation through each layer
    td = tau_df_fn(K_b_fn, dlai)

    # Preallocate solution arrays
    nz = lai.size
    nb = leaf_t.size
    I_dr = np.zeros((nz, nb))
    I_df_d = np.zeros_like(I_dr)
    I_df_u = np.zeros_like(I_dr)

    for i in range(nb):  # waveband loop

        # Preallocate equation arrays
        a = np.zeros((2*nz,))
        # a = np.zeros((2*(nz+1),))
        b = np.zeros_like(a)
        c = np.zeros_like(a)
        d = np.zeros_like(a)

        # Soil: upward flux
        a[0] = 0
        b[0] = 1
        c[0] = -albsoid[i]
        d[0] = swskyb[i] * tbcum[0] * albsoib[i]

        # Soil: downward flux
        refld = (1 - td[0]) * rho[i]
        trand = (1 - td[0]) * tau[i] + td[0]
        aiv = refld - trand * trand / refld
        biv = trand / refld
        a[1] = -aiv
        b[1] = 1
        c[1] = -biv
        d[1] = swskyb[i] * tbcum[0] * (1 - tb[0]) * (tau[i] - rho[i] * biv)

        # Inner canopy layers (excluding toc LAI=0 layer)
        # for j in range(nz - 1):
        for j in range(nz - 2):

            # Equation indices for this layer
            ju = 2 * (j + 1)
            jd = ju + 1

            # Upward flux
            refld = (1 - td[j]) * rho[i]
            trand = (1 - td[j]) * tau[i] + td[j]
            fiv = refld - trand * trand / refld
            eiv = trand / refld
            a[ju] = -eiv
            b[ju] = 1
            c[ju] = -fiv
            d[ju] = swskyb[i] * tbcum[j+1] * (1 - tb[j]) * (rho[i] - tau[i] * eiv)

            # Downward flux
            refld = (1 - td[j+1]) * rho[i]
            trand = (1 - td[j+1]) * tau[i] + td[j+1]
            aiv = refld - trand * trand / refld
            biv = trand / refld
            a[jd] = -aiv
            b[jd] = 1
            c[jd] = -biv
            d[jd] = swskyb[i] * tbcum[j+1+1] * (1 - tb[j+1]) * (tau[i] - rho[i] * biv)

        # Top canopy layer: upward flux
        refld = (1 - td[-1]) * rho[i]
        trand = (1 - td[-1]) * tau[i] + td[-1]
        fiv = refld - trand * trand / refld
        eiv = trand / refld
        a[-2] = -eiv
        b[-2] = 1
        c[-2] = -fiv
        d[-2] = swskyb[i] * tbcum[-1] * (1 - tb[-1]) * (rho[i] - tau[i] * eiv)

        # Top canopy layer: downward flux
        a[-1] = 0
        b[-1] = 1
        c[-1] = 0
        d[-1] = swskyd[i]

        # Solve system
        # res = tdma(a, b, c, d)
        import scipy
        res = scipy.linalg.solve(scipy.sparse.diags([a[1:], b, c[:-1]], [-1, 0, 1]).toarray(), d[:,np.newaxis]).squeeze()

        # Grab solutions
        swup = res[::2]   # upward flux above layer
        swdn = res[1::2]  # downward flux above layer (onto layer)

        # Store
        I_dr[:,i] = swskyb[i] * tbcum
        I_df_d[:,i] = swdn
        I_df_u[:,i] = swup

    return {
        "I_dr": I_dr,
        "I_df_d": I_df_d,
        "I_df_u": I_df_u,
        "F": I_dr / np.cos(psi) + 2 * I_df_d + 2 * I_df_u,
    }
