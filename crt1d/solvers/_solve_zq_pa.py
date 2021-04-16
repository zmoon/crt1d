"""
Modifications
-------------
(wrt. the verion in pyAPES)

* remove unneeded figure stuff

*

"""
import numpy as np

from .common import K_df_fn
from .common import tau_df_fn

#: machine epsilon
EPS = np.finfo(float).eps

short_name = "ZQ-pA"
long_name = "Zhao & Qualls multi-scattering (pyAPES)"


def solve_zq_pa(
    *,
    psi,
    I_dr0_all,
    I_df0_all,
    lai,
    clump,
    leaf_t,
    leaf_r,
    soil_r,
    K_b_fn,
):
    """


    Original notes from Samuli:

    .. code:: none

       OUTPUT:
           SWbo: direct SW at z (Wm-2(ground))
           SWdo: downwelling diffuse at z (Wm-2(ground))
           SWuo: upwelling diffuse at z (Wm-2(ground))
           Q_sl: incident SW normal to sunlit leaves (Wm-2)
           Q_sh: incident SW normal to shaded leaves (Wm-2)
           q_sl: absorbed SW by sunlit leaves (Wm-2(leaf))
           q_sh: absorbed SW by shaded leaves (Wm-2(leaf))
           q_soil: absorbed SW by soil surface (Wm-2(ground))
           f_sl: sunlit fraction of leaves (-): Note: to get sunlit fraction below
               all vegetation f_sl[0] / Clump
           alb: canopy albedo (-)
       USES:
           kbeam(ZEN,x) & kdiffuse(LAI,x=1) for computing beam and diffuse attenuation coeff
       SOURCE:
           Zhao W. & Qualls R.J. (2005). A multiple-layer canopy scattering model
           to simulate shortwave radiation distribution within a homogenous plant
           canopy. Water Resources Res. 41, W08409, 1-16.
       NOTE:
           At least for conifers NIR LeafAlbedo has to be decreased from leaf-scale  values to
           correctly model canopy albedo of clumped canopies.
           Adjustment from ~0.7 to 0.55 seems to be sufficient. This corresponds roughlty to
           a=a_needle*[4*STAR / (1- a_needle*(1-4*STAR))], where a_needle is needle albedo
           and STAR silhouette to total area ratio of a conifer shoot. STAR ~0.09-0.21 (mean 0.14)
           for Scots pine (Smolander, Stenberg et al. papers)
       CODE:
       Samuli Launiainen, Luke. Converted from Matlab & tested 15.5.2017
    """
    # original and computational grid
    # // LAI = Clump*sum(LAIz)  # effective LAI, corrected for clumping (m2 m-2)
    # // Lo = Clump*LAIz  # effective layerwise LAI (or PAI) in original grid
    # // Lcumo = np.cumsum(np.flipud(Lo), 0)  # cumulative plant area index from canopy top
    # // Lcumo = np.flipud(Lcumo)  # node 0 is canopy bottom, N is top

    # instead, assume that effective *cumulative* LAI has been passed in;
    # use Clump only for the absorption stuff
    LAI = lai[0]  # (total) effective LAI, corrected for clumping (m2 m-2)
    Clump = clump
    # dlai = lai[1:] - lai[:-1]
    dlai = np.r_[lai[1:] - lai[:-1], 0]  # add 0 to the top so dlai is same size as lai
    Lo = dlai  # layerwise LAI
    # `lai` input

    # solar zenith angle
    ZEN = psi

    # cumulative plant area index: node 0 is canopy bottom, N is top
    # this is the same as the `lai` input
    Lcumo = lai  # [:-1]  # neglect top LAI value of 0

    # --- create computation grid
    N = np.size(Lcumo)  # nr of original layers
    M = np.minimum(100, N)  # nr of comp. layers
    L = (
        np.ones([M + 2]) * LAI / M
    )  # effective leaf-area density, divided equally amongst layers (m2m-3)
    L[0] = 0.0
    L[M + 1] = 0.0

    # black leaf extinction coefficients for direct beam and diffuse radiation
    # Kb = kbeam(ZEN, x)
    # Kd = kdiffuse(LAI, x)

    # Kb and Kd using the existing machinery (passing in K_b_fn)
    Kb = K_b_fn(psi)
    Kd = K_df_fn(K_b_fn, LAI)

    # allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all = np.zeros(s)
    # I_df_d_ss_all = np.zeros(s)
    # I_df_u_ss_all = np.zeros(s)
    # F_ss_all      = np.zeros(s)

    for ib in range(nbands):
        # get irradiances in current band
        IbSky = I_dr0_all[ib]
        IdSky = I_df0_all[ib]

        # --- check inputs and create local variables
        # // IbSky = max(IbSky, 0.0001)
        # // IdSky = max(IdSky, 0.0001)
        # ^ this can skew results (irradiance in a band often less than 0.0001)
        #   instead we check inputs before passing in here

        # get optical param values in the band
        beta_L = leaf_r[ib]
        tau_L = leaf_t[ib]
        LeafAlbedo = beta_L + tau_L  # combined scattering
        # alpha_L = 1 - LeafAlbedo  # absorption
        SoilAlbedo = soil_r[ib]

        # ---- optical parameters
        aL = np.ones([M + 2]) * (1 - LeafAlbedo)  # leaf absorptivity
        tL = (
            np.ones([M + 2]) * tau_L / LeafAlbedo
        )  # transmission as fraction of scattered radiation
        rL = np.ones([M + 2]) * beta_L / LeafAlbedo  # reflection as fraction of scattered radiation

        # soil surface, no transmission
        aL[0] = 1.0 - SoilAlbedo
        tL[0] = 0.0
        rL[0] = 1.0

        # upper boundary = atm. is transparent for SW
        aL[M + 1] = 0.0
        tL[M + 1] = 1.0
        rL[M + 1] = 0.0

        # cumulative plant area from top
        # even though doesn't change with wavelength, need to define here since it is modified later
        Lcum = np.cumsum(np.flipud(L), 0)
        # ^ 0, ..., LAI, LAI

        # fraction of sunlit & shad ground area in a layer (-)
        # even though doesn't change with wavelength, need to define here since it is modified later
        f_sl = np.flipud(np.exp(-Kb * (Lcum)))
        # defined this way, f_sl[0] == 1

        # beam radiation at each layer
        Ib = f_sl * IbSky

        # propability for beam and diffuse penetration through layer without interception
        taub = np.zeros([M + 2])
        taud = np.zeros([M + 2])
        taub[0 : M + 2] = np.exp(-Kb * L)
        # taud[0:M+2] = np.exp(-Kd*L)  # < original expression
        taud[0 : M + 2] = tau_df_fn(K_b_fn, LAI / M)  # there is LAI/M in each layer
        # ^ not equivalent to original expression!

        # soil surface is non-transparent
        taub[0] = 0.0
        taud[0] = 0.0

        # backward-scattering functions (eq. 22-23) for beam rb and diffuse rd
        rb = np.zeros([M + 2])
        rd = np.zeros([M + 2])
        rb = 0.5 + 0.3334 * (rL - tL) / (rL + tL) * np.cos(ZEN)
        rd = 2.0 / 3.0 * rL / (rL + tL) + 1.0 / 3.0 * tL / (rL + tL)

        rb[0] = 1.0
        rd[0] = 1.0
        rb[M + 1] = 0.0
        rd[M + 1] = 0.0

        # --- set up tridiagonal matrix A and solve SW without multiple scattering
        # from A*SW = C (Zhao & Qualls, 2006. eq. 39 & 42)
        A = np.zeros([2 * M + 2, 2 * M + 2])

        # lowermost rows: 0 = soil surface, M+2 is upper boundary
        A[0, 0] = 1.0

        A[1, 0] = -(taud[1] + (1 - taud[1]) * (1 - aL[1]) * (1 - rd[1]))
        A[1, 1] = -1 * (taud[1] + (1 - taud[1]) * (1 - aL[1]) * (1 - rd[1])) * (1 - aL[0])
        A[1, 2] = 1 - 1 * rd[1] * (1 - aL[0]) * 1 * (1 - aL[1]) * (1 - taud[1])

        # middle rows
        for k in range(1, M + 1):
            A[2 * k - 1, 2 * k - 2] = -(taud[k] + (1 - taud[k]) * (1 - aL[k]) * (1 - rd[k]))
            A[2 * k - 1, 2 * k - 1] = (
                -rd[k - 1]
                * (taud[k] + (1 - taud[k]) * (1 - aL[k]) * (1 - rd[k]))
                * (1 - aL[k - 1])
                * (1 - taud[k - 1])
            )
            A[2 * k - 1, 2 * k] = 1 - rd[k - 1] * rd[k] * (1 - aL[k - 1]) * (1 - taud[k - 1]) * (
                1 - aL[k]
            ) * (1 - taud[k])

            A[2 * k, 2 * k - 1] = 1 - rd[k] * rd[k + 1] * (1 - aL[k]) * (1 - taud[k]) * (
                1 - aL[k + 1]
            ) * (1 - taud[k + 1])
            A[2 * k, 2 * k] = (
                -rd[k + 1]
                * (taud[k] + (1 - taud[k]) * (1 - aL[k]) * (1 - rd[k]))
                * (1 - aL[k + 1])
                * (1 - taud[k + 1])
            )
            A[2 * k, 2 * k + 1] = -(taud[k] + (1 - taud[k]) * (1 - aL[k]) * (1 - rd[k]))

        # uppermost node2*M+2
        A[2 * M + 1, 2 * M + 1] = 1.0
        del k
        # print A

        # --- RHS vector C
        C = np.zeros([2 * M + 2, 1])

        # lowermost row
        C[0] = SoilAlbedo * Ib[0]
        # C[0] = SoilAlbedo*Ib[-1]
        n = 1  # dummy
        for k in range(1, M + 1):  # k=2:M-1,
            C[n] = (
                (
                    1
                    - rd[k - 1]
                    * rd[k]
                    * (1 - aL[k - 1])
                    * (1 - taud[k - 1])
                    * (1 - aL[k])
                    * (1 - taud[k])
                )
                * rb[k]
                * (1 - taub[k])
                * (1 - aL[k])
                * Ib[k]
            )
            C[n + 1] = (
                (
                    1
                    - rd[k]
                    * rd[k + 1]
                    * (1 - aL[k])
                    * (1 - taud[k])
                    * (1 - aL[k + 1])
                    * (1 - taud[k + 1])
                )
                * (1 - taub[k])
                * (1 - aL[k])
                * (1 - rb[k])
                * Ib[k]
            )
            # Ib(k+1) Ib(k):n sijaan koska tarvitaan kerrokseen tuleva
            n = n + 2

        # uppermost row
        C[2 * M + 1] = IdSky

        # ---- solve A*SW = C
        SW = np.linalg.solve(A, C)

        # upward and downward hemispherical radiation (Wm-2 ground)
        SWu0 = SW[0 : 2 * M + 2 : 2]
        SWd0 = SW[1 : 2 * M + 2 : 2]
        del A, C, SW, k

        # ---- Compute multiple scattering, Zhao & Qualls, 2005. eq. 24 & 25.
        # downwelling diffuse after multiple scattering, eq. 24
        SWd = np.zeros([M + 1])
        for k in range(M - 1, -1, -1):  # downwards from layer k+1 to layer k: M-1..0
            X = SWd0[k + 1] / (
                1
                - rd[k]
                * rd[k + 1]
                * (1 - aL[k])
                * (1 - taud[k])
                * (1 - aL[k + 1])
                * (1 - taud[k + 1])
            )
            Y = (
                SWu0[k]
                * rd[k + 1]
                * (1 - aL[k + 1])
                * (1 - taud[k + 1])
                / (
                    1
                    - rd[k]
                    * rd[k + 1]
                    * (1 - aL[k])
                    * (1 - taud[k])
                    * (1 - aL[k + 1])
                    * (1 - taud[k + 1])
                )
            )
            SWd[k + 1] = X + Y
        SWd[0] = SWd[1]  # SWd0[0]
        # print SWd

        # upwelling diffuse after multiple scattering, eq. 25
        SWu = np.zeros([M + 1])
        for k in range(0, M, 1):  # upwards from layer k to layer k+1: 0..M-1
            X = SWu0[k] / (
                1
                - rd[k]
                * rd[k + 1]
                * (1 - aL[k])
                * (1 - taud[k])
                * (1 - aL[k + 1])
                * (1 - taud[k + 1])
            )
            Y = (
                SWd0[k + 1]
                * rd[k]
                * (1 - aL[k])
                * (1 - taud[k])
                / (
                    1
                    - rd[k]
                    * rd[k + 1]
                    * (1 - aL[k])
                    * (1 - taud[k])
                    * (1 - aL[k + 1])
                    * (1 - taud[k + 1])
                )
            )
            SWu[k] = X + Y
        SWu[M] = SWu[M - 1]

        # match dimensions of all vectors
        Ib = Ib[1 : M + 2]
        f_sl = f_sl[1 : M + 2]
        Lcum = np.flipud(Lcum[0 : M + 1])
        aL = aL[0 : M + 1]

        # --- NOW return values back to the original grid
        f_slo = np.exp(-Kb * (Lcumo))
        SWbo = f_slo * IbSky  # Beam radiation

        # interpolate diffuse fluxes
        X = np.flipud(Lcumo)
        xi = np.flipud(Lcum)
        SWdo = np.flipud(np.interp(X, xi, np.flipud(SWd)))
        SWuo = np.flipud(np.interp(X, xi, np.flipud(SWu)))
        del X, xi

        # incident radiation on sunlit and shaded leaves Wm-2
        # Q_sh = Clump * Kd * (SWdo + SWuo)  # normal to shaded leaves is all diffuse
        # Q_sl = Kb * IbSky + Q_sh  # normal to sunlit leaves is direct and diffuse

        # absorbed components
        aLo = np.ones(Lcumo.size) * (1 - LeafAlbedo)
        aDiffo = aLo * Kd * (SWdo + SWuo)
        aDiro = aLo * Kb * IbSky

        # stand albedo
        # // alb = SWuo[-1] / (IbSky + IdSky + EPS)
        alb = SWuo[-1] / (IbSky + IdSky)
        # print alb
        # soil absorption (Wm-2 (ground))
        q_soil = (1 - SoilAlbedo) * (SWdo[0] + SWbo[0])

        # correction to match absorption-based and flux-based albedo, relative error <3% may occur
        # in daytime conditions, at nighttime relative error can be larger but fluxes are near zero
        # // aa = (sum(aDiffo*Lo + aDiro*f_slo*Lo) + q_soil) / (IbSky + IdSky + EPS)
        aa = (sum(aDiffo * Lo + aDiro * f_slo * Lo) + q_soil) / (IbSky + IdSky)
        F = (1.0 - alb) / aa
        # print('F', F)
        if F <= 0 or np.isfinite(F) is False:
            F = 1.0

        aDiro = F * aDiro
        aDiffo = F * aDiffo
        q_soil = F * q_soil

        # sunlit fraction in clumped foliage; clumping means elements shade each other
        f_slo = Clump * f_slo

        # Following adjustment is required for energy conservation, i.e. total absorbed radiation
        # in a layer must equal difference between total radiation(SWup, SWdn) entering and leaving the layer.
        # Note that this requirement is not always fullfilled in canopy rad. models.
        # now sum(q_sl*f_slo* + q_sh*(1-f_slo)*Lo = (1-alb)*(IbSky + IdSky)
        # q_sh = aDiffo * Clump  # shaded leaves only diffuse
        # q_sl = q_sh + aDiro  # sunlit leaves diffuse + direct

        # original return:
        # return SWbo, SWdo, SWuo, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_slo, alb

        # store this band's results
        I_dr_all[:, ib] = SWbo
        I_df_u_all[:, ib] = SWuo
        I_df_d_all[:, ib] = SWdo
        F_all[:, ib] = SWbo / np.cos(psi) + 2 * SWuo + 2 * SWdo
        # TODO: add the absorption ones as extra outputs

    return {
        "I_dr": I_dr_all,
        "I_df_d": I_df_d_all,
        "I_df_u": I_df_u_all,
        "F": F_all,
    }
