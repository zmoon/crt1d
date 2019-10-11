"""
Calculations from the CRT solutions
"""

import numpy as np
from .solvers import res_keys_all_schemes

def E_to_PFD(E, wl_um):
    """Energy flux (W/m^2) to photon flux (umol photon / m^2 / s)
    """
    h = 6.626e-34  # J s
    c = 3.0e8  # m/s
    N_A = 6.022e23  # #/mol

    wl = wl_um[np.newaxis,:] * 1e-6  # um -> m
    
    e_wl_1 = h*c/wl  # J (per one photon)
    e_wl_mol = e_wl_1 * N_A  # J/mol
    e_wl_umol = e_wl_mol * 1e-6  # J/mol -> J/umol

    return E/e_wl_umol  


# TODO: maybe want to create and return as xr.Dataset instead
# or like sympl, dict of xr.Dataarrays 
def calc_leaf_absorption(cd, crs, 
        band_names_to_calc=['PAR', 'solar']):
    """Calculate layerwise leaf absorption from the CRT solver solutions
    
    for the state (one time/case)
    """

    if band_names_to_calc == 'all':
        band_names_to_calc = ['PAR', 'solar', 'NIR', 'UV']

    lai = cd['lai']
    dlai = cd['dlai']
    K_b = crs['K_b']
    G = crs['G']  # fractional leaf area projected in direction psi
    K_b = crs['K_b']  # G/cos(psi)

    leaf_r = cd['leaf_r']
    leaf_t = cd['leaf_t']
    leaf_a = 1 - (leaf_r + leaf_t)  # leaf element absorption coeff

    wl     = crs['wl']
    dwl    = crs['dwl']
    I_dr   = crs['I_dr']
    I_df_d = crs['I_df_d']
    I_df_u = crs['I_df_u']
    I_d = I_dr + I_df_d  # direct+diffuse downward irradiance

    try:
        wl_l = crs['wl_l']
        wl_r = crs['wl_r']
    except KeyError:
        wl_l = wl  # just use the wavelength we have for defining the bands
        wl_r = wl

    f_sl_interfaces = np.exp(-K_b*lai)  # fraction of sunlit
    f_sl = f_sl_interfaces[:-1] + 0.5*np.diff(f_sl_interfaces)
    f_sh = 1 - f_sl

    nlev = lai.size

    i = np.arange(nlev-1)
    ip1 = i + 1
    a = I_dr[ip1]-I_dr[i] + I_df_d[ip1]-I_df_d[i] + \
        I_df_u[i]-I_df_u[ip1]
    # ^ layerwise absorption by leaves in all bands
    #   inputs - outputs
    #   actual, not per unit LAI

    I_dr0 = I_dr[-1,:][np.newaxis,:]
    # a_dr =  I_dr0 * (K_b*f_sl*dlai)[:,np.newaxis] * leaf_a
    a_dr =  I_dr0 * (1 - np.exp(-K_b*f_sl*dlai))[:,np.newaxis] * leaf_a
    # ^ direct beam absorption (sunlit leaves only by definition)
    #
    # ** technically should be computed with exp **
    #   1 - exp(-K_b*L)
    # K_b*L is an approximation for small L
    # e.g., 
    #   K=1, L=0.01, 1-exp(-K*L)=0.00995

    a_df = a - a_dr  # absorbed diffuse radiation
    a_df_sl = a_df*f_sl[:,np.newaxis]
    a_df_sh = a_df*f_sh[:,np.newaxis]
    a_sl = a_df_sl + a_dr
    a_sh = a_df_sh
    assert( np.allclose(a_sl+a_sh, a) )

    #> photon flux density
    a_pfd = E_to_PFD(a, wl)


    #> absorbance in specific bands
    # isPAR = (wl >= 0.4) & (wl <= 0.7)
    # isNIR = (wl >= 0.7) & (wl <= 2.5)
    # isUV  = (wl >= 0.01) & (wl <= 0.4)
    # issolar = (wl >= 0.3) & (wl <= 5.0)
    isPAR = (wl_l >= 0.4) & (wl_r <= 0.7)
    isNIR = (wl_l >= 0.7) & (wl_r <= 2.5)
    isUV  = (wl_l >= 0.01) & (wl_r <= 0.4)
    issolar = (wl_l >= 0.3) & (wl_r <= 5.0)
    # also should have the edges somewhere
    # and read from that to make these
    # TODO: strictly should use left and right band definitions to do this,
    #       wl1 and wl2

    #> these are for the spectral version (/dwl)
    # a_PAR = si.simps(Sa[:,isPAR], wl[isPAR])

    #> non-spectral, just need to sum
    # a_PAR = a[:,isPAR].sum(axis=1)

    # isbands = [isPAR, issolar]
    # band_names = ['PAR', 'solar']

    bands = {
        'PAR' : isPAR,
        'solar' : issolar,
        'UV' : isUV,
        'NIR' : isNIR,
    }

    to_int = {
        'aI' : a,
        'aI_df' : a_df,
        'aI_dr' : a_dr,
        'aI_sh' : a_sh,
        'aI_sl' : a_sl,
        'aI_df_sl' : a_df_sl,
        'aI_df_sh' : a_df_sh,
        'I_d' : I_d,
        'I_dr' : I_dr,
        'I_df_d' : I_df_d,
        'I_df_u' : I_df_u,
    }

    res_int = {}
    for i, bn in enumerate(band_names_to_calc):
        isband = bands[bn]

        # res_int.update({
        #     f'{k}_{bn}': v[:,isband].sum(axis=1),
        #          for k, v in to_int.items()
        # })

        for k, v in to_int.items():
            k_int = f'{k}_{bn}'
            res_int[k_int] = v[:,isband].sum(axis=1)
            new_base = k.replace('I', 'PFD')  # don't want to replace NIR though!
            k_int_pfd = f'{new_base}_{bn}'
            # print(k, E_to_PFD(v[:,isband], wl[isband]).shape)
            res_int[k_int_pfd] = E_to_PFD(v[:,isband], wl[isband]).sum(axis=1)


    # res = {
    #     'aI' : a,
    #     'aI_sl': a_sl, 
    #     'aI_sh': a_sh,
    #     'aI_PAR': a_PAR,
    #     'I_d_PAR': I_d_PAR,
    #     'I_dr_PAR': I_dr_PAR,
    #     'I_df_d_PAR': I_df_d_PAR,
    #     'I_df_u_PAR': I_df_u_PAR,
    # }

    to_int_new = {k: v for k, v in to_int.items()
        if k not in res_keys_all_schemes}
    res = {**to_int_new, **res_int}

    return res


# TODO: fn to parse string varname to get dims/coords, units, and long_name
# need to move from model.to_xr and write better way (re or pyparsing?)

