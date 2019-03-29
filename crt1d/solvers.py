"""
These solvers are written as functions that can be called outside the model,
by supplying the necessary arguments using the cnpy_rad_state and cnpy_descrip 
dictionary inputs.


For now leaving all as looping over all available wavelengths. 

"""

import numpy as np
import scipy.integrate as si
import scipy.optimize as so
from scipy.sparse.linalg import spsolve  # used for Z&Q model only (so far)





def tau_b_fn(K_b_fn, psi, lai_val):
    """Transmittance of direct beam through the medium (foliage elements)

    We need to be able to integrate over different zenith angles (to calulate tau_df), 
    so we supply K_b_fn instead of K_b

    K_b_fn 
        := G_fn(psi)/cos(psi) where G_fn is "big G" for the chosen leaf angle distribution function.
    psi:
        Solar zenith angle (radians)
    lai_val: 
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
    return 2 * si.quad(f, 0, np.pi/2, epsrel=1e-9)[0]





#------------------------------------------------------------------------------------------------
# bl -- Beer--Lambert 
def solve_bl(cnpy_rad_state, cnpy_descrip):
    """Beer--Lambert solution based on Campbell (1986)
    
    but with some slight modifications to diffuse treatment. 

    Required inputs (passed in the dicts cnpy_rad_state, cnpy_descrip)
    ---------------
    I_dr0, Idf0: 
        direct and diffuse irradiances (W/m^2) at top of canopy
    mu: 
        cosine of the Solar zenith angle (SZA; radians)
    K_b: 
        black leaf extinction coeff
    wl, dwl: 
        wavelength band center and bandwidth (m)
    leaf_t, leaf_r:
        leaf element transmissivity and reflectivity (unitless)
    green:
        canopy green-ness factor (0, 1], to weight the leaf_t,_r


    Returns (in dict cnpy_rad_state)
    -------
    I_dr_all, I_df_d_all, I_df_u_all, F_all
        direct, downward diffuse, upward diffuse, actinic flux 
        (W/m^2, not W/m^2/m!)

    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    #psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']


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
    
    # pre-allocate arrays in which to save the solutions for each band
    I_dr_all = np.zeros((lai.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)

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

    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    # return I_dr_all, I_df_d_all, I_df_u_all, F_all 
    # return cnpy_rad_state  # shouldn't be strictly necessary but whatevs
    return



#------------------------------------------------------------------------------------------------
# 2s -- Dickinson-Sellers 2-stream
def solve_2s(cnpy_rad_state, cnpy_descrip):
    """Dickinson-Sellers 2-stream solution
    Most common scheme used in regional/climate models.

    Following Sellers (1983) and Dickinson (1985)
        mainly Dickinson -- variable names chosen to match his
    and including correction in later Dickinson paper
    
    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    orient = cnpy_descrip['orient']
    G_fn = cnpy_descrip['G_fn']
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']
    soil_r = cnpy_descrip['soil_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    #psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    #K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']

    theta_bar = np.deg2rad(mean_leaf_angle)  # mean leaf inclination angle; eq. 3

    # mu_bar := average inverse diffuse optical depth, per unit leaf area; p. 1336
    #   sa: angle of scattered flux
    mu_bar  = si.quad(lambda sa: np.cos(sa) / G_fn(sa) * -np.sin(sa), np.pi/2, 0)[0]  # p. 1336
    mu_bar2 = si.quad(lambda mu_prime: mu_prime / G_fn(np.arccos(mu_prime)), 0, 1)[0]
    assert( mu_bar == mu_bar2 )

    # where does this stuff with Pi come from ??? 
    # Dickinson 1983 ?
    if orient > 1:
        bigpi = (1 - orient**-2)**0.5  # the big pi
        theta = 1 + (np.log( (1+bigpi) / (1-bigpi) ) / (2 * bigpi * orient**2))  # theta
    else:
        bigpi = (1 - orient**2)**0.5;
        theta = (1 + np.arcsin(bigpi) ) / (orient * bigpi)  #
    muave = theta * orient / (orient + 1)  # mu_bar for spherical canopy? we don't seem to need this...

    L_T = lai[0]  # total LAI

    K = K_b  # for black leaves; should grey leaf version, K_b * k_prime, be used ???

    I_dr_all = np.zeros((lai.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)

    #
    #> run for each band individually
    #
    for i, band_width in enumerate(dwl):  

        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i] * band_width  # W / m^2
        I_df0 = I_df0_all[i] * band_width

        # load canopy optical properties
        alpha = green * leaf_r[i]  # leaf element reflectance
        tau   = green * leaf_t[i]  # leaf element transmittance
        rho_s = soil_r[i]         # soil reflectivity

        omega = alpha + tau  # scattering coefficient := omega = alpha + tau

        # diffuse beam upscatter param; eq. 3
        beta = ( 0.5 * ( alpha + tau + (alpha - tau) * np.cos(theta_bar)**2) ) / omega

        # single scattering albeo a_s; Table 2, p. 1339
        #   strictly should use the ellipsoidal version, but the orientation/eccentricity param is ~ 1,
        #   so spherical is good approx, and the form is much simpler
        a_s = omega/2 * ( 1 - mu * np.log( (mu + 1) / mu) )

        # direct beam upscatter param; eq. 4
        beta_0 = (1 + mu_bar * K ) / ( omega * mu_bar * K ) * a_s


        # ----------------------------------------------------------------------------
        # intermediate params used in calculation of solns

        b = (1 - (1 - beta)*omega)
        c = omega * beta
        d = omega * mu_bar * K * beta_0
        f = omega * mu_bar * K * (1 - beta_0)
        h = (b**2 - c**2)**0.5 / mu_bar
        sigma = (mu_bar * K)**2 + c**2 - b**2

        u1 = b - c / rho_s
        u2 = b - c * rho_s
        u3 = f + c * rho_s
        S1 = np.exp(-h * L_T)
        S2 = np.exp(-K * L_T)
        p1 = b + mu_bar * h
        p2 = b - mu_bar * h
        p3 = b + mu_bar * K
        p4 = b - mu_bar * K
        D1 = p1 * (u1 - mu_bar * h) * S1**-1 - p2 * (u1 + mu_bar * h) * S1
        D2 = (u2 + mu_bar * h) * S1**-1 - (u2 - mu_bar * h) * S1

        h1 = - d * p4 - c * f
        h2 = D1**-1 * ( (d - h1 / sigma * p3) * (u1 - mu_bar * h) * S1**-1          \
                        - p2 * (d - c - h1 / sigma * (u1 + mu_bar * K) ) * S2       \
                    )
        h3 = -D1**-1  * ( (d - h1 / sigma * p3) * (u1 + mu_bar * h) * S1           \
                        - p1 * (d - c - h1 / sigma * (u1 + mu_bar * K) ) * S2    \
                        )
        h4 = -f * p3 - c * d
        h5 = -D2**-1 * ( h4 / sigma * (u2 + mu_bar * h) / S1 \
                        + (u3 - h4 / sigma * (u2 - mu_bar * K) ) * S2 \
                    )
        h6 = D2**-1  * ( h4 / sigma * (u2 - mu_bar * h) * S1 \
                        + (u3 - h4 / sigma * (u2 - mu_bar * K) ) * S2 \
                    )
        h7 =  c / D1 * (u1 - mu_bar * h) / S1
        h8 = -c / D1 * (u1 + mu_bar * h) * S1
        h9 =   D2**-1  * (u2 + mu_bar * h) / S1
        h10 = -D2**-1  * (u2 - mu_bar * h) * S1


        # ----------------------------------------------------------------------------
        # direct beam soln ???
        #   contributions to downward and upward diffuse by scattering of direct radiation by leaves

        # I feel like this should work vector-ly without needing to use vectorize
        # maybe it is the dividing by sigma that is the issue
        # I know I wouldn't have done this if there wasn't one
        I_df_u_dr_fn = np.vectorize( lambda L: h1 * np.exp(-K * L) / sigma + h2 * np.exp(-h * L) + h3 * np.exp(h * L) )
        I_df_d_dr_fn = np.vectorize( lambda L: h4 * np.exp(-K * L) / sigma + h5 * np.exp(-h * L) + h6 * np.exp(h * L) )


        # ----------------------------------------------------------------------------
        # diffuse soln ???
        #   contributions to downward and upward diffuse by attenuation/scattering of top-of-canopy diffuse

        I_df_u_df_fn = np.vectorize( lambda L: h7 * np.exp(-h * L) +  h8 * np.exp(h * L) )
        I_df_d_df_fn = np.vectorize( lambda L: h9 * np.exp(-h * L) + h10 * np.exp(h * L) )


        # apply them fns
        I_df_u_dr = I_df_u_dr_fn(lai) * I_dr0
        I_df_d_dr = I_df_d_dr_fn(lai) * I_dr0
        I_df_u_df = I_df_u_df_fn(lai) * I_df0
        I_df_d_df = I_df_d_df_fn(lai) * I_df0

        # combine the contributions
        I_df_u = I_df_u_dr + I_df_u_df
        I_df_d = I_df_d_dr + I_df_d_df


        # Beer--Lambert direct beam attenuation
        I_dr = I_dr0 * np.exp(-K * lai)


        # save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d

    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    # return cnpy_rad_state
    return


#------------------------------------------------------------------------------------------------
# 4s -- 4-stream
def solve_4s(cnpy_rad_state, cnpy_descrip,
             mu_s=0.501):
    """4-stream from Tian et al. (2007) (featuring Dickinson)

    all eq./p. references in the code are to Tian et al. 2007 unless otherwise noted

    note that the authors use I for radiance, and F for irradiance (or actinic flux?)
    to be consistent with the other model codes I am using
        I for irradiance
        R for radiance
        F for actinic flux

    Model parameters
    ----------------
    mu_s: 
        Cosine of the dividing angle theta_s for the two beams: mu_s = cos(theta_s)
        Divides the hemisphere into 4 regions [-1, -mu_s], [-mu_s, 0], [0, mu_s], [mu_s, 1]
            two sectors, defined by [0, mu_s] and [mu_s, 1]
        Potential values:
            0.501: value suggested by the authors (~ 60 deg., arccos(0.501)*180/pi = 59.93)
            0.33998, 0.86114:  Gauss–Legendre quadrature points for n=4
                ~ 70 deg., ~ 31 deg. 
                Tian et al. (2007) and Li and Dobbie (1998) find the former to give more accurate results

        For 4-stream to provide benefits, the two beams should be diff angles from the vertical.
        (and I'm pretty sure that they are: technically one is zenith angle 0, the other at an angle to the zenith)


    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    #mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    #orient = cnpy_descrip['orient']
    G_fn = cnpy_descrip['G_fn']
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']
    soil_r = cnpy_descrip['soil_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    #K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']


    def eqns(x, y, d, direct=1):
        """Contribution to diffuse streams by scattering of direct/diffuse beam

        x: cumulative LAI
        y: [I2d, I1d, I1u, I2u]
        d: dictionary with various info needed for computation of the params
        direct: 1 means direct beam, 0 for diffuse beam
            for diffuse beam, we want the last term of the equations eliminated

        """

        G       = d['G']  # note that these could be class attrs instead of using dict

        mu_0    = d['mu_0']
        mu_1    = d['mu_1']
        mu_2    = d['mu_2']

        alpha_p = d['alpha_p']
        alpha_m = d['alpha_m']
        beta_p  = d['beta_p']
        beta_m  = d['beta_m']

        gamma_p = d['gamma_p']
        gamma_m = d['gamma_m']
        eps_p1  = d['eps_p1']
        eps_m1  = d['eps_m1']
        eps_p2  = d['eps_p2']
        eps_m2  = d['eps_m2']
        k_p1    = d['k_p1']
        k_m1    = d['k_m1']
        k_p2    = d['k_p2']
        k_m2    = d['k_m2']

        rhs1 = 1/mu_2 * ( (alpha_p - k_m2)*y[0] + beta_p*y[1] + beta_m*y[2] + alpha_m*y[3] ) + \
                direct * G * eps_m2 / mu_2 * np.exp(-G*x/mu_0)

        rhs2 = 1/mu_1 * ( beta_p*y[0] + (gamma_p - k_m1)*y[1] + gamma_m*y[2] + beta_m*y[3] ) + \
                direct * G * eps_m1 / mu_1 * np.exp(-G*x/mu_0)

        rhs3 = 1/mu_1 * ( -beta_m*y[0] - gamma_m*y[1] - (gamma_p - k_p1)*y[2] - beta_p*y[3] ) - \
                direct * G * eps_p1 / mu_1 * np.exp(-G*x/mu_0)

        rhs4 = 1/mu_2 * ( -alpha_m*y[0] - beta_m*y[1] - beta_p*y[2] - (alpha_p - k_p2)*y[3] ) - \
                direct * G * eps_p2 / mu_2 * np.exp(-G*x/mu_0)

        return np.vstack((rhs1, rhs2, rhs3, rhs4))



    def dfdr_bcs(ya, yb, d, direct=1, R0=0):
        """Boundary conditions
        eq. 8a ...

        direct: 1 for direct, 0 for diffuse version
            though that could be moved to be part of this fn instead

        """

        R_at_top = R0  # incident radiance (direct or diffuse)

        if direct == 1:
            R2d_top = 0  # note: I used for radiance instead of R in Tian et al. 2007
            R1d_top = 0

        elif direct == 0:
            R2d_top = R_at_top  # these should be radiance values, and we are assuming isotropic, so same values. y/n?
            R1d_top = R_at_top


        rho = d['rho']  # soil reflectance

        G       = d['G']
        LAI     = d['LAI']

        mu_0    = d['mu_0']
        mu_1    = d['mu_1']
        mu_2    = d['mu_2']

        # soil albedo conditions; ref eq. 8b
        I_down = 2 * np.pi * (mu_1*yb[1] + mu_2*yb[0])  # irradiance at bottom of canopy
        R_reflect = rho/np.pi * (I_down + direct * mu_0 * np.pi * R_at_top * np.exp(-G*LAI/mu_0) )

        # bcs
    #    return np.vstack((ya[0] - I2d_top,    # down beams at top of canopy
    #                      ya[1] - I1d_top,
    #                      yb[2] - I_reflect,  # up beams at soil surface (x = LAI)
    #                      yb[3] - I_reflect))
        return np.array([ya[0] - R2d_top,   ya[1] - R1d_top,
                            yb[2] - R_reflect, yb[3] - R_reflect])



    # ----------------------------------------------------------------------------------------------
    # Begin implementation

    K = K_b  # for black leaves; should grey leaf version, K_b * k_prime, be used ???

    LAI = lai[0]  # total LAI
    P = 1  # probably phase function ???, 1 indicates isotropic scattering <-- note: not exactly satisfied in the leaf optical data (refl doesn't always = transmit)
    G = G_fn(psi)
    G_int_1 = si.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), 0, mu_s)[0]  # Barr code did not do these integrals, assumed a constant G fn
    G_int_2 = si.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), mu_s, 1)[0]



    I_dr_all = np.zeros((lai.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)

    for i, band_width in enumerate(dwl):  # run for each band individually

    #    if i > 20:
    #        break

        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i] * band_width  # W / m^2
        I_df0 = I_df0_all[i] * band_width

        # convert to radiance, called "intensity" in the paper (Tian et al. 2007)
        #   ref for the conversion: Madronich 1987
        #   L often used for radiance, but in canopy stuff L is used for cumulative LAI so I use R
        R_dr0 = I_dr0 / (np.pi * mu)
        R_df0 = I_df0 / np.pi

        # soil albedo/reflectance
        rho = soil_r[i]

        # leaf optical props
        #   should the fact that omega isn't always 2*t = 2*r be accounted for ?
        #   could this information be used to form a more accurate phase function,
        #   since we know that the one direction (reflection or transmission) is often preffered (reflection!) ?
        r     = green * leaf_r[i]  # leaf element reflectance
        t     = green * leaf_t[i]  # leaf element transmittance
        omega = r + t               # single-scattering albedo


        # ------------------------------------------------------
        # calculate params; ref eqs. 4
        # note: assuming constant P for now, so many params have the same value
        #   should move the integration outside loop and store the value since it is the same

        mu_1    = 0.5 * mu_s**2
        mu_2    = 0.5 * (1 - mu_s**2)
        alpha_p = 0.5 * omega * P * (1 - mu_s) * G_int_2
        alpha_m = alpha_p
        beta_p  = 0.5 * omega * P * (1 - mu_s) * G_int_1
        beta_m  = beta_p
        gamma_p = 0.5 * omega * P * (mu_s - 0) * G_int_1
        gamma_m = gamma_p
        eps_p1  = 0.25 * omega * R_dr0 * P * (mu_s - 0)
        eps_m1  = eps_p1
        eps_p2  = 0.25 * omega * R_dr0 * P * (1 - mu_s)
        eps_m2  = eps_p2
        k_p1    = G_int_1
        k_m1    = G_int_1  # note Barr code did not have negative here. This seems to be right thing to do. at least needs abs()
        k_p2    = G_int_2
        k_m2    = G_int_2



        # create dict for input into the fns for the BVP solultion process
        #   probably not currently necessary since these vars are global within the model fn
        #   but would be helpful if redesign
        d = {'G': G, 'LAI': LAI,
                'rho': rho,
                'mu_0': mu, 'mu_1': mu_1, 'mu_2': mu_2, 'mu_s': mu_s,
                'alpha_p': alpha_p, 'alpha_m': alpha_m,
                'beta_p': beta_p, 'beta_m': beta_m,
                'gamma_p': gamma_p, 'gamma_m': gamma_m,
                'eps_p1': eps_p1, 'eps_m1': eps_m1, 'eps_p2': eps_p2, 'eps_m2': eps_m2,
                'k_p1': k_p1, 'k_m1': k_m1, 'k_p2': k_p2, 'k_m2': k_m2}


        # ------------------------------------------------------
        # contribution to diffuse due to scattering of direct beam

        x = np.linspace(0, LAI, 50)
        y0 = np.ones((4, x.size))  # initial guess

        fun = lambda x, y: eqns(x, y, d, direct=1)
        bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=1, R0=R_dr0)

        res = si.solve_bvp(fun, bcs, x, y0, tol=1e-6)

        y = res.sol(lai)  # solution splined over LAI vals

        # compute irradiances for the 4 streams from the radiance ("intensity") solution
        I2d_dr = 2 * np.pi * mu_2 * y[0,:]
        I1d_dr = 2 * np.pi * mu_1 * y[1,:]
        I1u_dr = 2 * np.pi * mu_1 * y[2,:]
        I2u_dr = 2 * np.pi * mu_2 * y[3,:]


        # ------------------------------------------------------
        # contribution to diffuse due to attenuation of canopy-incident diffuse

    #    x = np.linspace(0, LAI, 50)
    #    y0 = np.ones((4, x.size))  # initial guess

        fun = lambda x, y: eqns(x, y, d, direct=0)
        bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=0,  R0=R_df0)

        res = si.solve_bvp(fun, bcs, x, y0, tol=1e-6)

        y = res.sol(lai)  # solution splined over LAI vals

        # compute irradiances for the 4 streams from the radiance ("intensity") solution
        # each stream has a brother region on the -mu side of the hemisphere => x2
        I2d_df = 2 * np.pi * mu_2 * y[0,:]
        I1d_df = 2 * np.pi * mu_1 * y[1,:]
        I1u_df = 2 * np.pi * mu_1 * y[2,:]
        I2u_df = 2 * np.pi * mu_2 * y[3,:]


        # ------------------------------------------------------
        # combine results and save

        I2d = I2d_dr + I2d_df
        I1d = I1d_dr + I1d_df
        I1u = I1u_dr + I1u_df
        I2u = I2u_dr + I2u_df

        # downward and upward diffuse
        I_df_u = I1u + I2u
        I_df_d = I1d + I2d

        # Beer--Lambert direct beam attenuation
        I_dr = I_dr0 * np.exp(-K * lai)

        # save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d


    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    # return cnpy_rad_state
    return



#------------------------------------------------------------------------------------------------
# zq -- Zhao & Qualls multi-scattering
def solve_zq(cnpy_rad_state, cnpy_descrip):
    """Zhao & Qualls model (feat. multiple-scattering correction)

    All refs are to Zhao & Qualls (2005) unless otherwise noted

    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    #mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    #orient = cnpy_descrip['orient']
    G_fn = cnpy_descrip['G_fn']
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']
    soil_r = cnpy_descrip['soil_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']


    dlai = np.diff(lai)



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

    #> transmittance of direct and diffuse light through one layer
    #  most dlai vals are the same, so just calculate one tau_i and tau_psi value for now
    #  using Cambell & Norman eq. 15.5 fns defined above
    dlai_mean = np.abs(np.mean(dlai[dlai != 0]))
    tau_i_mean = tau_df_fn(K_b_fn, dlai_mean)
    tau_b_mean = tau_b_fn(K_b_fn, psi, dlai_mean)


#        LAI = lai[0]  # total LAI
#        G = G_fn(psi)


    #> lai subset??
    #  since diffuse at top level is inflated above the measured value otherwise...
    lai_plot = np.copy(lai)
    #lai = lai[:-1]  # upper boundary included in model


    I_dr_all = np.zeros((lai_plot.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    I_df_d_ss_all = np.zeros_like(I_dr_all)
    I_df_u_ss_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)
    F_ss_all = np.zeros_like(I_dr_all)

    for i, band_width in enumerate(dwl):  # run for each band individually

    #    if i > 20:
    #        break

        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i] * band_width  # W / m^2
        I_df0 = I_df0_all[i] * band_width


        # soil albedo/reflectance
        rho = soil_r[i]
        alpha0 = 1 - rho

        # leaf optical props; ref. p. 6, eq. 22
        beta_L  = green * leaf_r[i]  # leaf element reflectance
        tau_L   = green * leaf_t[i]  # leaf element transmittance
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

        C[0] = rho * S[0] #rho * S.min()#rho * I_dr0
        C[2*li-1] =   (1 - r[li-1]*r[li] * (1 - a[li-1]) * (1 - t[li-1]) * (1 - a[li]) * (1 - t[li])) * \
                    r_psi * (1 - t_psi) * (1 - a[li]) * S
        C[2*li] = (1 - r[li]*r[li+1] * (1 - a[li]) * (1 - t[li]) * (1 - a[li+1]) * (1 - t[li+1])) * \
                    (1 - t_psi) * (1 - a[li]) * (1 - r_psi) * S
        C[-1] = I_df0  # top-of-canopy diffuse


        # ------------------------------------------
        # find soln to A x = C, where x is the radiation flux density (irradiance, hopefully)
        #   maybe it is supposed to actinic flux, since the term "flux density" is used
        #
        # note: could also use a Thomas algorithm solver method (write a fn for it)
        #

        x = spsolve(A, C)
        SWu0 = x[::2]  # "original downward and upward hemispherical shortwave radiation flux densities"
        SWd0 = x[1::2] # i.e., before multiple scattering within layers is accounted for


        # ------------------------------------------
        # multiple scattering correction
        # p. 6, eqs. 24, 25

        SWd = np.zeros((m+1, ))
        SWu = np.zeros_like(SWd)

        # note that li = 1:m

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

    #    S = np.append(S[::-1], I_dr0)
    #    I_df_d_ss_all[:,i] = SWd0  # single-scattering results
    #    I_df_d_all[:,i] =     SWd  # after multiple-scattering corrections
    #    I_df_u_ss_all[:,i] = SWu0  # single-scattering results
    #    I_df_u_all[:,i] =     SWu  # after multiple-scattering corrections
    #    F_ss_all[:,i] = S / mu + 2 * SWu0 + 2 * SWd0
    #    F_all[:,i] =    S / mu +  2 * SWu +  2 * SWd

        I_dr_all[:,i] = I_dr0 * np.exp(-K * lai)


    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    return



#------------------------------------------------------------------------------------------------
# bf -- Bodin & Franklin improved Goudriaan
def solve_bf(cnpy_rad_state, cnpy_descrip):
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
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    #mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    #orient = cnpy_descrip['orient']
    #G_fn = cnpy_descrip['G_fn']
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']
    soil_r = cnpy_descrip['soil_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    #psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    #K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']


    lai_tot = lai[0]
    assert(lai_tot == lai.max())


    #> following variables in B&F.
    #  except I for irradiance, instead of the R B&F uses for...
    
    k_b = K_b  # direct beam attenuation coeff
    
    I_dr_all = np.zeros((lai.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)

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
        
        #> save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr/mu + 2*I_df_u + 2*I_df_d

    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    return


#------------------------------------------------------------------------------------------------
# gd -- Goudriaan
def solve_gd(cnpy_rad_state, cnpy_descrip):
    """
    Goudriaan (1977) scheme, 
    according to Bodin and Franklin (2012)

    """

    #
    #> Get canopy description and radiation parameters that we need
    #
    #L = cnpy_descrip['L']  # total LAI
    lai = cnpy_descrip['lai']  # (cumulative) LAI profile
    #mean_leaf_angle = cnpy_descrip['mean_leaf_angle']  # (deg.)
    #orient = cnpy_descrip['orient']
    #G_fn = cnpy_descrip['G_fn']
    green = cnpy_descrip['green']  # canopy green-ness factor
    leaf_t = cnpy_descrip['leaf_t']
    leaf_r = cnpy_descrip['leaf_r']
    soil_r = cnpy_descrip['soil_r']

    I_dr0_all = cnpy_rad_state['I_dr0_all']
    I_df0_all = cnpy_rad_state['I_df0_all']
    #psi = cnpy_rad_state['psi']
    mu = cnpy_rad_state['mu']
    K_b = cnpy_rad_state['K_b']
    #K_b_fn = cnpy_rad_state['K_b_fn']
    wl = cnpy_rad_state['wl']
    dwl = cnpy_rad_state['dwl']


    lai_tot = lai[0]
    assert(lai_tot == lai.max())


    #> following variables in B&F.
    #  except I for irradiance, instead of the R B&F uses for...
    
    k_b = K_b  # direct beam attenuation coeff
    
    I_dr_all = np.zeros((lai.size, wl.size))
    I_df_d_all = np.zeros_like(I_dr_all)
    I_df_u_all = np.zeros_like(I_dr_all)
    F_all = np.zeros_like(I_dr_all)

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
        
        #> scattered radiation (one stream only)
        #  B&F eq. 5
        I_sc = I_dr0*(1-rho_c)*np.exp(-k_prime*k_b*lai) + \
                -I_dr0*(1-sigma)*np.exp(-k_b*lai)
                
        #  assuming up/down scattered from leaves equal for now..
        I_sc_d = 0.5*I_sc
        I_sc_u = 0.5*I_sc
        
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
        #  assuming up/down scattered from leaves equal for now..
        I_df_d = I_sc_d + I_df
        I_df_u = I_sc_u + I_sr
        
        #> save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr/mu + 2*I_df_u + 2*I_df_d

    #> update values in cnpy_rad_state
    cnpy_rad_state['I_dr_all'] = I_dr_all
    cnpy_rad_state['I_df_d_all'] = I_df_d_all
    cnpy_rad_state['I_df_u_all'] = I_df_u_all
    cnpy_rad_state['F_all'] = F_all

    return












#------------------------------------------------------------------------------------------------
# compile available schemes

available_schemes = [\
    dict(ID='bl', solver=solve_bl, shortname='B–L', longname='Beer–Lambert'),
    dict(ID='2s', solver=solve_2s, shortname='2s',  longname='Dickinson–Sellers two-stream'),
    dict(ID='4s', solver=solve_4s, shortname='4s',  longname='four-stream'),
    dict(ID='zq', solver=solve_zq, shortname='ZQ',  longname='Zhao & Qualls multi-scattering'),
    dict(ID='bf', solver=solve_bf, shortname='BF',  longname='Bodin & Franklin improved Goudriaan'),
    dict(ID='gd', solver=solve_gd, shortname='Gou', longname='Goudriaan'),
    ]


















