"""
Calculations from the CRT solutions
"""

import numpy as np
from .solvers import res_keys_all_schemes

def e_to_photon(E, wl_um):
    """Energy flux (W/m^2) to photon flux (umol photon / m^2 / s) ?
    """
    h = 6.626e-34
    c = 3.0e8
    N_A = 6.022e23

    wl = wl_um * 1e-6  # um -> m
    
    e_wl_1 = h*c/wl
    e_wl_mol = e_wl_1 * N_A

    return e_wl_mol * 1e6  # mol -> umol


# TODO: maybe want to create and return as xr.Dataset instead
# or like sympl, dict of xr.Dataarrays 
def calc_leaf_absorption(cd, crs, 
        band_names_to_calc=['PAR', 'solar']):
    """Calculate layerwise leaf absorption from the CRT solver solutions
    
    for the state (one time/case)
    """

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

    #> absorbance in specific bands
    isPAR = (wl >= 0.4) & (wl <= 0.7)
    isNIR = (wl >= 0.7) & (wl <= 2.5)
    isUV  = (wl >= 0.01) & (wl <= 0.4)
    issolar = (wl >= 0.3) & (wl <= 5.0)
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

        res_int.update({
            f'{k}_{bn}': v[:,isband].sum(axis=1)
                 for k, v in to_int.items()
        })


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

