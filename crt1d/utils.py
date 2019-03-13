

import numpy as np

from .leaf_area_alloc import canopy_lai_dist


def distribute_lai(cdd, n):
    """Create LAI profile based on input descriptors of the leaf area distribution.

    Inputs
    ------
    cdd:  canopy description dict. contains many things. see the default CSV for example
    n:    number of layers we want

    """
    LAI = cdd['lai_tot']  # total canopy LAI
    h_bottom = cdd['h_bot'][-1]  # canopy bottom is the bottom of the bottom-most layer
    
    #> form inputs to the layer class
    nlayers = len(cdd['lai_frac'])
    layers = []
    for i in range(nlayers):
        layer = dict(h_max=cdd['h_max_lad'][i],
                     h_top=cdd['h_top'][i], 
                     lad_h_top=cdd['lad_h_top'][i],
                     fLAI=cdd['lai_frac'][i])
        layers.append(layer)
    
    
    cld = canopy_lai_dist(h_bottom, layers[::-1], LAI)  # form dist of leaf area

#    h_canopy = cld.lds[-1]['h_top']  # canopy height is the top of the upper-most layer
    h_canopy = cdd['h_canopy']

    dlai = float(LAI) / (n - 1)  # desired const lai increment
#    print dlai

    lai = np.zeros((n, ))
    z = h_canopy * np.ones_like(lai)  # bottoms of layers?

    ub = h_canopy
    LAIcum = 0
    for i in range(n-2, -1, -1):  # build from top down

        if LAI - LAIcum < dlai:
            assert( i == 0 )
            z[0] = h_bottom
            lai[0] = LAI

        else:
            lb = cld.inv_cdf(ub, dlai)
            z[i] = lb
            lai[i] = lai[i+1] + dlai

            ub = lb
            LAIcum += dlai

    return lai, z


def load_canopy_descrip(fname):
    """Load items from CSV file and return as dict
    The file should use the same fieldnames as in the default one!
    Pandas might also do this.
    """
    varnames, vals = np.genfromtxt(fname, 
                                   usecols=(0, 1), skip_header=1, delimiter=',',  
                                   dtype=str, unpack=True)
    
    n = varnames.size
    d_raw = {varnames[i]: vals[i] for i in range(n)}
#    print d_raw
    
    d = d_raw  # could copy instead
    for k, v in d.items():
        if ';' in v:            
            d[k] = [float(x) for x in v.split(';')]
        else:
            d[k] = float(v)
#    print d
    
    #> checks
    assert( np.array(d['lai_frac']).sum() == 1.0 )
 
    return d
    
    


def load_spectral_props(DOY, hhmm):
    """
    returns: wl, dwl, I_dr0, I_df0, leaf_r, leaf_t, soil_r

    radiation data for top-of-canopy must already exist

    """

    # ----------------------------------------------------------------------------------------------
    # radiation data from SPCTRAL2

    spectral_data_file = \
        '{base:s}/../SPCTRAL2_xls/Borden_DOY{DOY:d}/EDT{hhmm:s}.csv'.format(base=this_dir, DOY=DOY, hhmm=hhmm)
    spectral_data = np.loadtxt(spectral_data_file,
                               delimiter=',', skiprows=1)
    wl = spectral_data[:,0]    # wavelength (um); can't use 'lambda' because it is reserved
    dwl = np.append(np.diff(wl), np.nan)  # um; UV region more important so add nan to end instead of beginning, though doesn't really matter since outside leaf data bounds
    # note: could also do wl band midpoints instead..
    I_dr0 = spectral_data[:,1]  # direct (spectral) irradiance impinging on canopy (W/m^2/um)
    I_df0 = spectral_data[:,2]  # diffuse

#    wl_a = 0.35  # lower limit of leaf data; um
    wl_a = 0.30  # lower limit of leaf data, extended; um
    wl_b = 2.6   # upper limit of leaf data; um
    leaf_data_wls = (wl >= wl_a) & (wl <= wl_b)

    # use only the region where we have leaf data
    #  initially, everything should be at the SPCTRAL2 wls
    wl    = wl[leaf_data_wls]
    dwl   = dwl[leaf_data_wls]
    I_dr0 = I_dr0[leaf_data_wls]
    I_df0 = I_df0[leaf_data_wls]

    # also eliminate < 0 values ??
    #   this was added to the leaf optical props generation
    #   but still having problem with some values == 0


    # ----------------------------------------------------------------------------------------------
    # soil reflectivity

    soil_r = np.ones_like(wl)
    soil_r[wl <= 0.7] = 0.1100  # this is the PAR value
    soil_r[wl > 0.7]  = 0.2250  # near-IR value


    # ----------------------------------------------------------------------------------------------
    # green leaf properties

    leaf_data_file = '{base:s}/../ideal-leaf-optical-props/leaf-idealized-rad_SPCTRAL2_wavelengths_extended.csv'.format(base=this_dir)
    leaf_data = np.loadtxt(leaf_data_file,
                           delimiter=',', skiprows=1)
    leaf_t = leaf_data[:,1][leaf_data_wls]
    leaf_r = leaf_data[:,2][leaf_data_wls]

    leaf_t[leaf_t == 0] = 1e-10
    leaf_r[leaf_r == 0] = 1e-10

    return wl, dwl, I_dr0, I_df0, leaf_r, leaf_t, soil_r


