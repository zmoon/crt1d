
import numpy as np

short_name = 'BF'
long_name = 'Bodin & Franklin improved Goudriaan'

def solve_bf(*, psi,
    I_dr0_all, I_df0_all, wl, dwl,
    lai,
    leaf_t, leaf_r, green, soil_r, 
    K_b_fn, 
    ):
    """
    Bodin and Franklin (2012)
    improved Goudriaan (1977) scheme (~ 1.5 streams)
    
    based on Saylor's ACCESS v3.0 code 
    in subroutine CalcRadProfiles of module CanopyPhysics
    
    Saylor's description:
        Uses algorithms of Bodin & Franklin (2012) Efficient modeling of sun/shade canopy radiation dynamics
        explicitly accounting for scattering, Geoscientific Model Development, 5, 535-541. 
        ... with elliptical kb from Campbell & Norman (1998) pp. 247-259
        ... clear-sky effective emissivity of Prata (1996) Q.J.R. Meteorol. Soc., 122, 1127-1151.
        ... cloudy-sky correction algorithm of Crawford and Duchon (1999) J. Appl. Meteorol., 48, 474-480.
        ... based on evaluations of Flerchinger et al. (2009) Water Resour. Res., 45, W03423, doi:10.1029/2008WR007394.
    

    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    # lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    # #mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    # #orient = cnpy_descrip['orient']
    # #G_fn = cnpy_descrip['G_fn']
    # green = cnpy_descrip['green']  # canopy green-ness factor
    # leaf_t = cnpy_descrip['leaf_t']
    # leaf_r = cnpy_descrip['leaf_r']
    # soil_r = cnpy_descrip['soil_r']

    # I_dr0_all = cnpy_rad_state['I_dr0_all']
    # I_df0_all = cnpy_rad_state['I_df0_all']
    # #psi = cnpy_rad_state['psi']
    # mu = cnpy_rad_state['mu']
    # K_b = cnpy_rad_state['K_b']
    # #K_b_fn = cnpy_rad_state['K_b_fn']
    # wl = cnpy_rad_state['wl']
    # dwl = cnpy_rad_state['dwl']


    K_b = K_b_fn(psi)
    mu = np.cos(psi)
    lai_tot = lai[0]
    assert(lai_tot == lai.max())


    #> following variables in B&F.
    #  except I for irradiance, instead of the R B&F uses for...
    
    k_b = K_b  # direct beam attenuation coeff
    
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

    for i, band_width in enumerate(dwl):  # run for each band individually

        #> calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i] * band_width  # W / m^2
        I_df0 = I_df0_all[i] * band_width

        #> relevant properties for the band (using values at waveband LHS)
        r_l = leaf_r[i]
        t_l = leaf_t[i]
        W = soil_r[i]  # ground-sfc albedo, assume equal to soil reflectivity
        sigma = green * (r_l + t_l)
        alpha = 1 - sigma  # absorbed by leaf
        k_prime = np.sqrt(alpha)  # bulk attenuation coeff for a leaf; Moneith & Unsworth eq. 4.16
        #K = K_b * k_prime  # approx extinction coeff for non-black leaves; ref Moneith & Unsworth p. 120
    
        #> (total) canopy reflectance
        #  Spitters (1986) eq. 1, based on Goudriaan 1977
        #    note k_prime = (1-sigma)^0.5 as defined above
        #    and mu = cos(psi) = sin(beta)
        rho_c = ((1-k_prime)/(1+k_prime)) * (2 / (1 + 1.6*mu))
    
        #> diffuse light attenuation coeff
        k_d = 0.8*np.sqrt(1-sigma)  # B&F eq. 2
        
        #> attenuation of incoming diffuse
        #  B&F eq. 1
        I_df = I_df0 * (1-rho_c) * np.exp(-k_d*lai)
        
        #> attenuation of direct beam due to absorption and scattering
        #
        I_dr = I_dr0 * np.exp(-k_b*lai)
        
        #> fraction of leaves / leaf area in the direct beam
        A_sl = np.exp(-k_b*lai)  # "fraction of sunlit leaves" B&F eq. 3
        
        #> downwelling scattered radiation (from direct beam)
        #  B&F eq. 8
        I_sc_d = I_dr0 * t_l * \
            ( (np.exp(-k_b*lai) - np.exp(-k_d*lai)) / (k_d - k_b) ) 

        #> upwelling scattered radiation (from direct beam)
        #  B&F eq. 9
        I_sc_u = I_dr0 * r_l * \
            ( (np.exp(-k_b*lai) - np.exp(+k_d*lai - (k_b+k_d)*lai_tot)) / (k_d + k_b) )

        #> total direct beam radiation scattered by foliage elements
        #  B&F eq. 10
        I_sc = I_sc_d + I_sc_u
        
        #> ground-sfc reflectance term (upward)
        #  B&F eq. 11
        # L_tot should correspond to index 0: `z[0]` is lowest level
        I_sr = W * (I_dr0*A_sl[0] + I_df[0] + I_sc_d[0]) * np.exp(-k_d * (lai_tot-lai))


        #> rad absorbed by shaded leaves
        #  B&F eq. 14
        I_sh_a = (1-A_sl) * \
            ( k_d/k_prime*I_df + k_d/np.sqrt(1-r_l)*I_sc_u + k_d/np.sqrt(1-t_l)*I_sc_d )

        #> rad absorbed by sunlit leaves (direct beam term added to the end)
        #  B&F eq. 15
        #  
        I_sl_a = A_sl * \
            ( k_d/k_prime*I_df + k_d/np.sqrt(1-r_l)*I_sc_u + k_d/np.sqrt(1-t_l)*I_sc_d + \
                k_b*I_dr0 )

        #> final downward and upward diffuse
        I_df_d = I_sc_d + I_df
        I_df_u = I_sc_u + I_sr
        
        #> check that the absorbed matches what we would expect
        # I_a = I_sh_a + I_sl_a
        # I_a2 = I_dr[]
        # j = np.arange(1, lai.size)
        # jm1 = j - 1
        # I_sh_a_2 = (1-A_sl[j])*( I_df_d[j] - I_df_d[jm1] + I_df_u[jm1] - I_df_u[j] )
        # I_sh_a_2 = ( I_df_d[j] - I_df_d[jm1] + I_df_u[jm1] - I_df_u[j] )
        j = np.arange(0, lai.size-1)
        jp1 = j + 1
        I_sh_a_2 = (1-A_sl[j])*( I_df_d[jp1] - I_df_d[j] + I_df_u[j] - I_df_u[jp1] )

        # print('1:', I_sh_a)
        # print('2:', I_sh_a_2)
        # assert(np.allclose( I_sh_a[1:], I_sh_a_2 ))

        #> save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr/mu + 2*I_df_u + 2*I_df_d


    return I_dr_all, I_df_d_all, I_df_u_all, F_all 
