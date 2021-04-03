# fmt: off
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve

from .common import tau_b_fn
from .common import tau_df_fn


short_name = 'ZQ'
long_name = 'Zhao & Qualls multi-scattering'


def solve_zq(
    *, psi,
    I_dr0_all, I_df0_all,
    lai,
    leaf_t, leaf_r, soil_r,
    K_b_fn, G_fn,
):
    """Zhao & Qualls model (feat. multiple-scattering correction).

    All refs are to Zhao & Qualls (2005) unless otherwise noted.

    Notes
    -----
    choice of nlayers has more of an impact with this scheme than some of the others
    higher number generally better

    """
    dlai = np.diff(lai)
    mu = np.cos(psi)
    K_b = K_b_fn(psi)

    # backward scattering functions
    def r_psi_fn(bl, tl, psi=psi):
        """Backscatter coeff for directional radiation; eq. 22
        """
        return 0.5 + 0.3334 * ((bl - tl)/(bl + tl)) * np.cos(psi)

    def r_fn(bl, tl):
        """Backscatter coeff for hemispherical (isotropic) rad.; eq. 23
        """
        return 2.0/3 * (bl / (bl + tl)) + 1.0/3 * (tl / (bl + tl))

    K = K_b  # for black leaves

    # Transmittance of direct and diffuse light through one layer
    # Most dlai vals are the same, so just calculate one tau_i and tau_psi value for now
    # using Cambell & Norman eq. 15.5 fns defined above
    dlai_mean = np.abs(np.mean(dlai[dlai != 0]))
    tau_i_mean = tau_df_fn(K_b_fn, dlai_mean)
    tau_b_mean = tau_b_fn(K_b_fn, psi, dlai_mean)

#        LAI = lai[0]  # total LAI
#        G = G_fn(psi)

    # lai subset??
    #   since diffuse at top level is inflated above the measured value otherwise...
    # // lai_plot = np.copy(lai)
    # // lai = lai[:-1]  # upper boundary included in model

    # Allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all   = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all      = np.zeros(s)
    I_df_d_ss_all = np.zeros(s)
    I_df_u_ss_all = np.zeros(s)
    F_ss_all      = np.zeros(s)

    for i in range(nbands):  # run for each band individually

        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i]  # W / m^2
        I_df0 = I_df0_all[i]

        # soil albedo/reflectance
        rho = soil_r[i]
        alpha0 = 1 - rho

        # leaf optical props; ref. p. 6, eq. 22
        beta_L  = leaf_r[i]  # leaf element reflectance
        tau_L   = leaf_t[i]  # leaf element transmittance
        alpha_L = 1 - (beta_L + tau_L)  # leaf element absorbance

        # ------------------------------------------
        # to form the matrix A we need:
        #   r_i: backward scattering fraction for layer
        #   tau_i: fraction of hemispherical shortwave radiation that totally penetrates layer (without encountering leaf)
        #     this should be equal to tau_d from Cambell & Norman (eq. 15.5)
        #   alpha_i: fraction of total intercepted radiation that is absorbed, within a layer
        #     this should be equal to the leaf element absorbance
        #
        # r, t, a varnames used instead to make it easier to read/type

        m = lai.size  # number of layers in the model

        r = r_fn(beta_L, tau_L) * np.ones((m+2, ))
        t = tau_i_mean * np.ones_like(r)  # really should calculate the value for each layer, but almost all are the same
        a = alpha_L * np.ones_like(r)

        # boundary values for imaginary bottom, top layers
        r[0], r[-1] = 1, 0  # at ground (z=0), no forward scattering, only back
        t[0], t[-1] = 0, 1  # no penetration into ground; 100% penetration through top ghost layer
        a[0], a[-1] = alpha0, 0  # 1 - rho absorbed at ground sfc; no absorption (by leaves) in top ghost layer

        A = np.zeros((2*m + 2, 2*m + 2))

        li = np.arange(m) + 1  # layer indices (plus 1) <- inds corresponding to the non-boundary values

        # now following p. 8
        A[0,0] = 1
        A[2*li-1,2*li-1-1] = -(t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li]))
        A[2*li-1,2*li+0-1] = -r[li-1] * (t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li])) * (1 - a[li-1]) * (1 - t[li-1])
        A[2*li-1,2*li+1-1] = 1 - r[li-1] * r[li] * (1 - a[li-1]) * (1 - t[li-1]) * (1 - a[li]) * (1 - t[li])
        A[2*li+1-1,2*li+0-1] = 1 - r[li] * r[li+1] * (1 - a[li]) * (1 - t[li]) * (1 - a[li+1]) * (1 - t[li+1])
        A[2*li+1-1,2*li+1-1] = -r[li+1] * (t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li])) * (1 - a[li+1]) * (1 - t[li+1])
        A[2*li+1-1,2*li+2-1] = -(t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li]))
        A[-1,-1] = 1

        # ------------------------------------------
        # to form C we need: (following p. 8 still)
        #   S: direct beam extinction
        #   r_psi

    #    S = I_dr0 * np.exp(-K * lai[::-1])
        S = I_dr0 * np.exp(-K * lai)
        r_psi = r_psi_fn(beta_L, tau_L)
        t_psi = tau_b_mean

        C = np.zeros((2*m+2, ))

        C[0] = rho * S[0]  # rho * S.min() # rho * I_dr0
        C[2*li-1] = (
            1 - r[li-1]*r[li] * (1 - a[li-1]) * (1 - t[li-1]) * (1 - a[li]) * (1 - t[li])
        ) * r_psi * (1 - t_psi) * (1 - a[li]) * S
        C[2*li] = (
            1 - r[li]*r[li+1] * (1 - a[li]) * (1 - t[li]) * (1 - a[li+1]) * (1 - t[li+1])
        ) * (1 - t_psi) * (1 - a[li]) * (1 - r_psi) * S
        C[-1] = I_df0  # top-of-canopy diffuse

        # ------------------------------------------
        # find soln to A x = C, where x is the radiation flux density (irradiance, hopefully)
        #   maybe it is supposed to actinic flux, since the term "flux density" is used
        #
        # note: could also use a Thomas algorithm solver method (write a fn for it)
        #

        # ---- standard solve
        # x = np.linalg.solve(A, C)

        # ---- solve using sparse matrix machinery
        # convert A to sparse matrix
        # A_sparse = dia_matrix((A, (-1, 0, 1)), shape=(3, A.shape[1]))
        A_sparse = dia_matrix(A)
        # print(A_sparse.shape)
        # print(A_sparse)
        # print(A_sparse.offsets, A_sparse.data.shape)
        assert tuple(A_sparse.offsets) == (-1, 0, 1)
        assert A_sparse.data.shape == (3, A.shape[1])
        x = spsolve(A_sparse.tocsr(), C)  # needs CSR or CSC format

        SWu0 = x[::2]   # "original downward and upward hemispherical shortwave radiation flux densities"
        SWd0 = x[1::2]  # i.e., before multiple scattering within layers is accounted for

        # ------------------------------------------
        # multiple scattering correction
        # p. 6, eqs. 24, 25

        SWd = np.zeros((m+1, ))
        # SWu = np.zeros_like(SWd)
        SWu = np.zeros((m+1, ))  # < please pylint

        # note that li = 1:m (including m)

        # eq. 24; i+1 -> li, i -> li - 1
        SWd[li] = SWd0[li] / (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li])) + \
            r[li] * (1 - a[li]) * (1 - t[li]) * SWu0[li-1] / \
            (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li]))

        # eq. 25
        SWu[li-1] = SWu0[li-1] / (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li])) + \
            r[li-1] * (1 - a[li-1]) * (1 - t[li-1]) * SWd0[li] / \
            (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li]))

        # -----------------------------------------
        # save

    #    I_df_d_ss_all[:,i] = SWd0[:-1]  # single-scattering results
    #    I_df_d_all[:,i] =     SWd[:-1]  # after multiple-scattering corrections
    #    F_ss_all[:,i] = S / mu + 2 * SWu0[1:] + 2 * SWd0[:-1]
    #    F_all[:,i] =    S / mu +  2 * SWu[1:] +  2 * SWd[:-1]

        I_df_d_ss_all[:,i] = SWd0[1:]  # single-scattering results
        I_df_d_all[:,i] =     SWd[1:]  # after multiple-scattering corrections
        I_df_u_ss_all[:,i] = SWu0[:-1]  # single-scattering results
        I_df_u_all[:,i] =     SWu[:-1]  # after multiple-scattering corrections
        F_ss_all[:,i] = S / mu + 2 * SWu0[:-1] + 2 * SWd0[1:]
        F_all[:,i] =    S / mu +  2 * SWu[:-1] +  2 * SWd[1:]

        # I_df_d_ss_all[:,i] = SWd0[-1:]  # single-scattering results
        # I_df_d_all[:,i] =     SWd[-1:]  # after multiple-scattering corrections
        # I_df_u_ss_all[:,i] = SWu0[1:]  # single-scattering results
        # I_df_u_all[:,i] =     SWu[1:]  # after multiple-scattering corrections
        # F_ss_all[:,i] = S / mu + 2 * SWu0[1:] + 2 * SWd0[:-1]
        # F_all[:,i] =    S / mu +  2 * SWu[1:] +  2 * SWd[:-1]

    #    S = np.append(S[::-1], I_dr0)
    #    I_df_d_ss_all[:,i] = SWd0  # single-scattering results
    #    I_df_d_all[:,i] =     SWd  # after multiple-scattering corrections
    #    I_df_u_ss_all[:,i] = SWu0  # single-scattering results
    #    I_df_u_all[:,i] =     SWu  # after multiple-scattering corrections
    #    F_ss_all[:,i] = S / mu + 2 * SWu0 + 2 * SWd0
    #    F_all[:,i] =    S / mu +  2 * SWu +  2 * SWd

        I_dr_all[:,i] = I_dr0 * np.exp(-K * lai)

    return dict(
        I_dr=I_dr_all,
        I_df_d=I_df_d_all,
        I_df_u=I_df_u_all,
        F=F_all,
        I_df_d_ss=I_df_d_ss_all,
        I_df_u_ss=I_df_u_ss_all,
        F_ss=F_ss_all
    )
