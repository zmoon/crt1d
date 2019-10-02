
import numpy as np

from . import input_data_dir
from .leaf_angle_dist import ( G_ellipsoidal_approx, 
    mean_leaf_angle_to_orient, )
from .leaf_area_alloc import ( distribute_lai_from_cdd, 
    distribute_lai_beta )
from .utils import ( load_canopy_descrip, load_default_leaf_soil_props, 
    load_default_toc_spectra )


cnpy_descrip_keys = [\
    'lai', 'z', 'dlai', 'lai_tot', 'lai_eff', 
    'mean_leaf_angle', 
    'green', 
    'leaf_t', 
    'leaf_r',
    'soil_r',
    'wl_leafsoil', 
    'orient',  # don't really need both this and mla as input
    'G_fn',
    'K_b_fn', 
    ]

cnpy_rad_state_input_keys = [\
    'I_dr0', 'I_df0',  # spectral ()
    'wl', 'dwl',  # for the toc spectra
    'psi', 'mu',
    'K_b', 
    ]

cnpy_rad_state_keys = cnpy_rad_state_input_keys + [\
    'I_dr', 'I_df_d', 'I_df_u', 'F',  # direct from schemes
    'a_PAR', 'a_solar', 'a_UV', 'a_spectral',  # calculated (some schemes provide)
    'a_PAR_sl', 'a_PAR_sh', #... median wl for energy and photon perspective, photon flux, ...
    ]

def load_default_case(nlayers):
    """Idealized beta leaf dist, 

    """
    h_c = 20.
    LAI = 4.
    lai, z = distribute_lai_beta(h_c, LAI, nlayers)

    #> spectral things
    #  includes: top-of-canopy BC, leaf reflectivity and trans., soil refl, 
    #  for now, assume wavelengths from the BC apply to leaf as well
    wl_default, dwl_default, \
    leaf_r_default, leaf_t_default, \
    soil_r_default = load_default_leaf_soil_props()

    mla = 57  # for spherical? (check)
    orient = mean_leaf_angle_to_orient(mla)
    G_fn = lambda psi_: G_ellipsoidal_approx(psi_, orient)

    cnpy_descrip = dict(\
        lai=lai, z=z,
        green=1.0,
        mean_leaf_angle=mla,
        clump=1.0,
        leaf_t=leaf_t_default,
        leaf_r=leaf_r_default,
        soil_r=soil_r_default,
        wl_leafsoil=wl_default,
        orient=orient,
        G_fn=G_fn,
        )

    wl, dwl, I_dr0, I_df0 = load_default_toc_spectra()

    sza = 20  # deg.
    psi = np.deg2rad(sza)
    # mu = np.cos(psi)

    cnpy_rad_state = dict(\
        I_dr0_all=I_dr0, I_df0_all=I_df0, wl=wl, dwl=dwl, 
        psi=psi,
        )


    return cnpy_descrip, cnpy_rad_state








def load_Borden95_default_case(nlayers):
    """ """
    cdd_default = load_canopy_descrip(input_data_dir+'/'+'default_canopy_descrip.csv')
    lai, z = distribute_lai_from_cdd(cdd_default, nlayers)




# if __name__ == '__main__':

#     from .__init__ import input_data_dir 

#     cd, crs = load_default_case(10)
#     print(cd)
#     print(crs)
