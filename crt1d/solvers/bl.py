
import numpy as np
import scipy.integrate as si

short_name = 'B-L'
long_name = 'Beer-Lambert'

def solve_bl(*, psi,
    I_dr0_all, I_df0_all, wl, dwl,
    lai,
    leaf_t, leaf_r, green, 
    K_b_fn, 
    ):
    """Beer--Lambert solution based on Campbell (1986)
    
    but with some slight modifications to diffuse treatment. 

    Required inputs
    ---------------
    psi : float
        Solar zenith angle (SZA; radians)
    I_dr0_all, Idf0_all : np array (n_cases, n_wl) 
        direct and diffuse irradiances (W/m^2) at top of canopy
        at all wavelengths
    wl, dwl : np array (n_wl)
        wavelength band centers and bandwidths (m)
        for the top-of-canopy irradiances and other spectral quantities
    lai : np array (n_z)
        LAI profile
        lai[0] = total LAI, lai[-1] = 0 (canopy-atmos interface layer)
    leaf_t, leaf_r : np array (n_wl)
        leaf element spectral transmissivity and reflectivity (unitless)
    green : float
        canopy green-ness factor (0, 1], to weight the leaf_t,_r
    K_b_fn : K_b_fn(psi) 
        the function to be used to compute K_b
        (depends on which leaf angle distribution is used)
    

    Returns
    -------
    I_dr_all, I_df_d_all, I_df_u_all, F_all
        direct, downward diffuse, upward diffuse, actinic flux 
        (W/m^2, not W/m^2/m!)

    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    # lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    # green = cnpy_descrip['green']  # canopy green-ness factor
    # leaf_t = cnpy_descrip['leaf_t']
    # leaf_r = cnpy_descrip['leaf_r']

    # I_dr0_all = cnpy_rad_state['I_dr0_all']
    # I_df0_all = cnpy_rad_state['I_df0_all']
    # #psi = cnpy_rad_state['psi']
    # mu = cnpy_rad_state['mu']
    # K_b = cnpy_rad_state['K_b']
    # K_b_fn = cnpy_rad_state['K_b_fn']
    # wl = cnpy_rad_state['wl']
    # dwl = cnpy_rad_state['dwl']

    # lai_tot = lai[0]
    # assert(lai_tot == lai.max())

    #
    #> calculate additional needed params
    #
    mu = np.cos(psi)
    K_b = K_b_fn(psi)  # could put this as default in fn args?

    #
    #> Functions for calculating K_df (diffuse light extinction coeff)
    #    ref. Campbell & Norman eq. 15.5
    #    these are the same for each band for a given canopy, 
    #    so the code could be placed outside this fn
    #
    #tau = lambda psi_, L: np.exp(-K_b_fn(psi_) * k_prime * L_)  # transmittance of beam, including transmission *through* leaves (downward scattering)
    tau_b = lambda psi_, L_: np.exp(-K_b_fn(psi_) * L_)

    def tau_df(lai_val):
        """Transmissivity for diffuse light
        Weighted hemispherical integral of direct beam transmissivity tau_b"""
        f = lambda psi_: tau_b(psi_, lai_val) * np.sin(psi_) * np.cos(psi_)
        return 2 * si.quad(f, 0, np.pi/2, epsrel=1e-9)[0]

    # construct LUT for diffuse extinction so integration is not redone for each band
    tau_df_lut = {'{:.6f}'.format(lai_val): tau_df(lai_val) for lai_val in set(lai)}
    K_df_fn = lambda L: -np.log(tau_df_lut['{:.6f}'.format(L)]) / L
    K_df_fn = np.vectorize(K_df_fn)
    
    #> allocate arrays in which to save the solutions for each band
    # I_dr_all = np.zeros((lai.size, wl.size))
    # I_df_d_all = np.zeros_like(I_dr_all)
    # I_df_u_all = np.zeros_like(I_dr_all)
    # F_all = np.zeros_like(I_dr_all)
    s = (lai.size, wl.size)  # to make pylint shut up until it supports _like()
    I_dr_all   = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all      = np.zeros(s)

    #
    #> run for each band individually
    #
    for i, band_width in enumerate(dwl):
        
        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i] * band_width  # W / m^2
        I_df0 = I_df0_all[i] * band_width

        # relevant properties for the band (using values at waveband LHS)
        # tranmission through leaf and reflection by leaf both treated as scattering processes
        scat = green * (leaf_t[i] + leaf_r[i])  # + (1 - green) * (other wavelength tr and rf)  ???
        alpha = 1 - scat  # absorbed by leaf; ref Moneith & Unsworth p. 47
        k_prime = np.sqrt(alpha)  # bulk attenuation coeff for a leaf; Moneith & Unsworth eq. 4.16

        K = K_b * k_prime  # approx extinction coeff for non-black leaves; ref Moneith & Unsworth p. 120

        # calculate profiles
        #   here I_df is just downward diffuse
        I_dr = I_dr0 * np.exp(-K_b * lai)
        I_df = I_df0 * np.exp(-1*K_df_fn(lai[:-1]) * lai[:-1])  # doesn't work for L=0 since L in denom
        I_df = np.append(I_df, I_df0)

        # approximate the contribution of scattering of the direct beam to diffuse irradiance within canopy
        #   using the grey leaf K
        I_df_dr = I_dr0 * (np.exp(-K * lai) - np.exp(-K_b * lai))

        # approximate the contribution to diffuse from scattered direct.
        #   assume 1/2 downward, 1/2 upward for now
        #   though a more appropriate fraction (downward vs upward) could be computed, following the Z&Q methods
        I_df += 0.5 * I_df_dr

        # save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df
        I_df_u_all[:,i] = I_df  # assume u and d equiv for now...
        F_all[:,i] = I_dr / mu + 2 * I_df  # actinic flux (upward + downward hemisphere components)


    return I_dr_all, I_df_d_all, I_df_u_all, F_all 

