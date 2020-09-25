import numpy as np

short_name = "Gou"
long_name = "Goudriaan"


def solve_gd(
    *,
    psi,
    I_dr0_all,
    I_df0_all,  # wl, dwl,
    lai,
    leaf_t,
    leaf_r,
    soil_r,
    K_b_fn,
):
    """
    Goudriaan (1977) scheme
    according to Bodin and Franklin (2012)

    """

    #
    # > Get canopy description and radiation parameters that we need
    #

    # > following variables in B&F.
    #  except I for irradiance, instead of the R B&F uses for...

    K_b = K_b_fn(psi)
    k_b = K_b  # direct beam attenuation coeff
    mu = np.cos(psi)
    lai_tot = lai[0]
    assert lai_tot == lai.max()

    # > allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all = np.zeros(s)
    aI_sl_all = np.zeros(s)
    aI_sh_all = np.zeros(s)

    for i in range(nbands):  # run for each band individually

        # > calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i]  # W / m^2
        I_df0 = I_df0_all[i]

        # > relevant properties for the band (using values at waveband LHS)
        r_l = leaf_r[i]
        t_l = leaf_t[i]
        W = soil_r[i]  # ground-sfc albedo, assume equal to soil reflectivity
        sigma = r_l + t_l
        alpha = 1 - sigma  # absorbed by leaf
        k_prime = np.sqrt(alpha)  # bulk attenuation coeff for a leaf; Moneith & Unsworth eq. 4.16
        # K = K_b * k_prime  # approx extinction coeff for non-black leaves; ref Moneith & Unsworth p. 120

        # > (total) canopy reflectance
        #  Spitters (1986) eq. 1, based on Goudriaan 1977
        #    note k_prime = (1-sigma)^0.5 as defined above
        #    and mu = cos(psi) = sin(beta)
        rho_c = ((1 - k_prime) / (1 + k_prime)) * (2 / (1 + 1.6 * mu))

        # > diffuse light attenuation coeff
        k_d = 0.8 * np.sqrt(1 - sigma)  # B&F eq. 2

        # > attenuation of incoming diffuse
        #  B&F eq. 1
        I_df = I_df0 * (1 - rho_c) * np.exp(-k_d * lai)

        # > attenuation of direct beam due to absorption and scattering
        #
        I_dr = I_dr0 * np.exp(-k_b * lai)

        # > fraction of leaves / leaf area in the direct beam
        A_sl = np.exp(-k_b * lai)  # "fraction of sunlit leaves" B&F eq. 3

        # > scattered radiation (one stream only)
        #  B&F eq. 5
        I_sc = I_dr0 * (1 - rho_c) * np.exp(-k_prime * k_b * lai) + -I_dr0 * (1 - sigma) * np.exp(
            -k_b * lai
        )

        #  assuming up/down scattered from leaves equal for now..
        I_sc_d = 0.5 * I_sc
        I_sc_u = 0.5 * I_sc

        # > ground-sfc reflectance term (upward)
        #  B&F eq. 11
        # L_tot should correspond to index 0: `z[0]` is lowest level
        I_sr = W * (I_dr0 * A_sl[0] + I_df[0] + I_sc_d[0]) * np.exp(-k_d * (lai_tot - lai))

        # > rad absorbed by shaded leaves
        #  B&F eq. 14
        I_sh_a = (1 - A_sl) * (
            k_d / k_prime * I_df + k_d / np.sqrt(1 - r_l) * I_sc_u + k_d / np.sqrt(1 - t_l) * I_sc_d
        )

        # > rad absorbed by sunlit leaves (direct beam term added to the end)
        #  B&F eq. 15
        #
        I_sl_a = A_sl * (
            k_d / k_prime * I_df
            + k_d / np.sqrt(1 - r_l) * I_sc_u
            + k_d / np.sqrt(1 - t_l) * I_sc_d
            + k_b * I_dr0
        )

        # > final downward and upward diffuse
        #  assuming up/down scattered from leaves equal for now..
        I_df_d = I_sc_d + I_df
        I_df_u = I_sc_u + I_sr

        # > save
        I_dr_all[:, i] = I_dr
        I_df_d_all[:, i] = I_df_d
        I_df_u_all[:, i] = I_df_u
        F_all[:, i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d
        aI_sl_all[:, i] = I_sl_a
        aI_sh_all[:, i] = I_sh_a

    # return I_dr_all, I_df_d_all, I_df_u_all, F_all
    return dict(
        I_dr=I_dr_all,
        I_df_d=I_df_d_all,
        I_df_u=I_df_u_all,
        F=F_all,
        aI_sl=aI_sl_all,
        aI_sh=aI_sh_all,
        aI=aI_sl_all + aI_sh_all,
    )
