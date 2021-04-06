import numpy as np

import crt1d as crt


def test_n79_against_bonan():
    # Set up same inputs as in Bonan case
    lap = crt.leaf_area.distribute_lai_beta_bonan(20, 6, 61)
    lai = lap.lai
    z = lap.z
    psi = 30 * (np.pi / 180)

    # PAR, NIR
    leaf_r = np.r_[0.1, 0.45]
    leaf_t = np.r_[0.05, 0.25]
    I_dr0_all = np.r_[0.8, 0.8]
    I_df0_all = np.r_[0.2, 0.2]
    soil_r = np.r_[0.1, 0.2]
    wl = np.r_[0.55, 1.6]
    dwl = np.r_[0.3, 1.8]
    # wle = np.r_[0.4, 0.7, 2.5]

    G_fn = crt.leaf_angle.G_spherical  # 0.5

    p = {
        "lai": lai,
        "z": z,
        "psi": psi,
        "leaf_r": leaf_r,
        "leaf_t": leaf_t,
        "soil_r": soil_r,
        "I_dr0_all": I_dr0_all,
        "I_df0_all": I_df0_all,
        "wl": wl,
        "wl_leafsoil": wl,
        "dwl": dwl,
        "clump": 1.0,
        "G_fn": G_fn,
    }

    ds = crt.Model(scheme="n79", **p).run(tau_d_method="9sky").calc_absorption().to_xr()

    # Load Bonan data
    # toc -> ground (initially, but we flip)
    # layer index, lai, sunlit absorbed PAR, shaded, sunlit absorbed NIR, shaded
    data0 = np.loadtxt(
        np.DataSource(None).open(  # None -> temporary dir
            "https://raw.githubusercontent.com/zmoon/bonanmodeling/master/sp_14_03/data.txt"
        ),
        skiprows=1,
    )[::-1]

    f_sl = ds.f_slm.values
    f_sh = 1 - f_sl

    y0s = [data0[:, 2], data0[:, 3], data0[:, 4], data0[:, 5]]
    ys = [
        ds.aI_sl.isel(wl=0) / (f_sl * ds.dlai),  # conv. to per unit (sunlit or shaded) leaf area
        ds.aI_sh.isel(wl=0) / (f_sh * ds.dlai),
        ds.aI_sl.isel(wl=1) / (f_sl * ds.dlai),
        ds.aI_sh.isel(wl=1) / (f_sh * ds.dlai),
    ]

    # Compute mean absolute error (mostly due to limited precision in the Bonan output file)
    maes = [np.abs(y - y0).mean() for y, y0 in zip(ys, y0s)]

    assert not any(mae > 1e-6 for mae in maes)
