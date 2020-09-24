import numpy as np

from .common import ( 
    # tau_b_fn as tau_b_fn_psi, 
    tau_df_fn,
)

short_name = 'B–L'
long_name = 'Beer–Lambert'

def solve_bl(*, psi,
    I_dr0_all, I_df0_all, #wl, dwl,
    lai,
    leaf_t, leaf_r, green, 
    K_b_fn, 
    ):
    """Beer--Lambert solution based on Campbell (1986)
    but with some slight modifications to diffuse treatment.

    This is (thus far) the simplest scheme included.
    """
    #
    #> calculate additional needed params
    #
    mu = np.cos(psi)
    K_b = K_b_fn(psi)  # could put this as default in fn args?

    #> transmission of direct beam
    tau_b = np.exp(-K_b * lai)

    #> transmission of hemispherical diffuse to each point in the LAI profile
    # tau_df = tau_df_fn(K_b_fn, lai)
    tau_df = np.zeros(lai.shape)
    for i in range(lai.size):
        tau_df[i] = tau_df_fn(K_b_fn, lai[i])

    #> allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all   = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all      = np.zeros(s)

    #
    #> run for each band individually
    #
    for i in range(nbands):
        
        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i]  # W / m^2
        I_df0 = I_df0_all[i]

        # relevant properties for the band (using values at waveband LHS)
        # tranmission through leaf and reflection by leaf both treated as scattering processes
        scat = green * (leaf_t[i] + leaf_r[i])  # + (1 - green) * (other wavelength tr and rf)  ???
        alpha = 1 - scat  # absorbed by leaf; ref Moneith & Unsworth p. 47
        k_prime = np.sqrt(alpha)  # bulk attenuation coeff for a leaf; Moneith & Unsworth eq. 4.16

        K = K_b * k_prime  # approx extinction coeff for non-black leaves; ref Moneith & Unsworth p. 120

        tau_g = np.exp(-K * lai)  # grey leaf transmission

        # calculate profiles
        #   here I_df is just downward diffuse
        I_dr = I_dr0 * tau_b
        I_df = I_df0 * tau_df

        # approximate the contribution of scattering of the direct beam to diffuse irradiance within canopy
        #   using the grey leaf K
        I_df_dr = I_dr0 * (tau_g - tau_b)

        # approximate the contribution to diffuse from scattered direct.
        #   assume 1/2 downward, 1/2 upward for now
        #   though a more appropriate fraction (downward vs upward) could be computed, following the Z&Q methods
        I_df += 0.5 * I_df_dr

        # TODO: get better upward diffuse. it is too high this way
        # use leaf_r to calculate single scattering from top layer?

        # save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df
        I_df_u_all[:,i] = 0 #I_df  # < don't technically have a good expression for upward diffuse currently
        # I_df_u_all[:,i] = 0.5*I_df_dr  # 
        F_all[:,i] = I_dr / mu + 2 * I_df  # actinic flux (upward + downward hemisphere components)
        # ^ upward diffuse (which is not good currently) here doesn't contribute to F

    # return I_dr_all, I_df_d_all, I_df_u_all, F_all 
    return dict(\
        I_dr = I_dr_all, 
        I_df_d = I_df_d_all, 
        I_df_u = I_df_u_all, 
        F = F_all
        )