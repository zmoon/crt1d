"""
Data provided with crt1d: default/sample spectra,
as well as functions that use external (not-required) libraries to compute more spectra.
"""
import numpy as np

from .. import DATA_BASE_DIR

DATA_DIR_STR = DATA_BASE_DIR.as_posix()


def load_default_leaf_soil_props():
    """Load arrays for the ideal green leaf and simplified (two-value) soil reflectivity."""
    # ----------------------------------------------------------------------------------------------
    # green leaf properties (alread at SPCTRAL2 wavelengths)

    leaf_data_file = "{data:s}/ideal-green-leaf_SPCTRAL2-wavelengths.csv".format(data=DATA_DIR_STR)
    leaf_data = np.loadtxt(leaf_data_file, delimiter=",", skiprows=1)
    wl = leaf_data[:, 0]
    # ^ all SPCTRAL2 wavelengths (to 4 um); assume band center (though may be left/edge...)
    dwl = np.append(np.diff(wl), np.nan)
    # ^ um; UV region more important so add nan to end instead of beginning, though doesn't really matter since outside leaf data bounds

    #    wl_a = 0.35  # lower limit of leaf data; um
    wl_a = 0.30  # lower limit of leaf data, extended; um
    wl_b = 2.6  # upper limit of leaf data; um
    leaf_data_wls = (wl >= wl_a) & (wl <= wl_b)

    # use only the region where we have leaf data
    #  initially, everything should be at the SPCTRAL2 wls
    wl = wl[leaf_data_wls]
    dwl = dwl[leaf_data_wls]

    leaf_t = leaf_data[:, 1][leaf_data_wls]
    leaf_r = leaf_data[:, 2][leaf_data_wls]

    leaf_t[leaf_t == 0] = 1e-10
    leaf_r[leaf_r == 0] = 1e-10
    # TODO: ^ also eliminate < 0 values ??
    #   this was added to the leaf optical props generation
    #   but still having problem with some values == 0 in certain schemes

    # ----------------------------------------------------------------------------------------------
    # soil reflectivity (assume these for now (used in Fuentes 2007)

    # soil_r = np.ones_like(wl)
    soil_r = np.ones(wl.shape)  # < please pylint
    soil_r[wl <= 0.7] = 0.1100  # this is the PAR value
    soil_r[wl > 0.7] = 0.2250  # near-IR value

    return wl, dwl, leaf_r, leaf_t, soil_r


def load_default_toc_spectra():
    """Load sample top-of-canopy spectra (direct and diffuse).

    The SPCTRAL2 irradiances are in spectral form: W m^-2 um^-1
    """
    fp = "{data:s}/SPCTRAL2_xls_default-spectrum.csv".format(data=DATA_DIR_STR)

    wl, I_dr0, I_df0 = np.loadtxt(fp, delimiter=",", skiprows=1, unpack=True)

    # TODO: update dwl
    dwl0 = np.diff(wl)
    dwl = np.r_[dwl0, dwl0[0]]

    wl_a = 0.30  # lower limit of leaf data, extended; um
    wl_b = 2.6  # upper limit of leaf data; um
    leaf_data_wls = (wl >= wl_a) & (wl <= wl_b)

    # use only the region where we have leaf data
    wl = wl[leaf_data_wls]
    dwl = dwl[leaf_data_wls]
    SI_dr0 = I_dr0[leaf_data_wls]  # [np.newaxis,:]  # expects multiple cases
    SI_df0 = I_df0[leaf_data_wls]  # [np.newaxis,:]

    return wl, dwl, SI_dr0 * dwl, SI_df0 * dwl
