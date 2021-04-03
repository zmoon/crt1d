import numpy as np
import scipy.integrate as integrate

short_name = "4s"
long_name = "four-stream"


def solve_4s(
    *, psi, I_dr0_all, I_df0_all, lai, leaf_t, leaf_r, soil_r, K_b_fn, G_fn, mu_s=0.501  # wl, dwl,
):
    r"""4-stream from Tian et al. (2007) (featuring Dickinson).

    Notes
    -----
    All eq./p. references in the code are to Tian et al. 2007 (:cite:`tian_four-stream_2007`)
    unless otherwise noted.

    Note that the authors use :math:`I` for radiance, and :math:`F` for irradiance.
    To be consistent with the other model codes, I am using

    * ``I`` for irradiance

    * ``R`` for radiance

    * ``F`` for actinic flux


    `mu_s` (:math:`\mu_s`) is the cosine of the dividing angle for the two beams:

    .. math::
       \mu_s = \cos(\theta_s)

    :math:`\mu_s` divides each hemisphere into two sectors:
    :math:`[0, \mu_s]` and :math:`[\mu_s, 1]` for the upward
    and :math:`[-1, -\mu_s]` and :math:`[-\mu_s, 0]` for the downward.

    Potential values for the `mu_s` "hyperparameter":

    * 0.501: value suggested by the authors (~ 60 deg., arccos(0.501)*180/pi = 59.93)

    * 0.33998, 0.86114: Gauss–Legendre quadrature points for n=4, ~ 70 deg., ~ 31 deg.
      Tian et al. (2007) and Li and Dobbie (1998) find (0.33998) the former to give more accurate results.
    """

    K_b = K_b_fn(psi)
    mu = np.cos(psi)

    def eqns(x, y, d, direct=1):
        """Contribution to diffuse streams by scattering of direct/diffuse beam

        x: cumulative LAI
        y: [I2d, I1d, I1u, I2u]
        d: dictionary with various info needed for computation of the params
        direct: 1 means direct beam, 0 for diffuse beam
            for diffuse beam, we want the last term of the equations eliminated

        """

        G = d["G"]  # note that these could be class attrs instead of using dict

        mu_0 = d["mu_0"]
        mu_1 = d["mu_1"]
        mu_2 = d["mu_2"]

        alpha_p = d["alpha_p"]
        alpha_m = d["alpha_m"]
        beta_p = d["beta_p"]
        beta_m = d["beta_m"]

        gamma_p = d["gamma_p"]
        gamma_m = d["gamma_m"]
        eps_p1 = d["eps_p1"]
        eps_m1 = d["eps_m1"]
        eps_p2 = d["eps_p2"]
        eps_m2 = d["eps_m2"]
        k_p1 = d["k_p1"]
        k_m1 = d["k_m1"]
        k_p2 = d["k_p2"]
        k_m2 = d["k_m2"]

        rhs1 = 1 / mu_2 * (
            (alpha_p - k_m2) * y[0] + beta_p * y[1] + beta_m * y[2] + alpha_m * y[3]
        ) + direct * G * eps_m2 / mu_2 * np.exp(-G * x / mu_0)

        rhs2 = 1 / mu_1 * (
            beta_p * y[0] + (gamma_p - k_m1) * y[1] + gamma_m * y[2] + beta_m * y[3]
        ) + direct * G * eps_m1 / mu_1 * np.exp(-G * x / mu_0)

        rhs3 = 1 / mu_1 * (
            -beta_m * y[0] - gamma_m * y[1] - (gamma_p - k_p1) * y[2] - beta_p * y[3]
        ) - direct * G * eps_p1 / mu_1 * np.exp(-G * x / mu_0)

        rhs4 = 1 / mu_2 * (
            -alpha_m * y[0] - beta_m * y[1] - beta_p * y[2] - (alpha_p - k_p2) * y[3]
        ) - direct * G * eps_p2 / mu_2 * np.exp(-G * x / mu_0)

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

        rho = d["rho"]  # soil reflectance

        G = d["G"]
        LAI = d["LAI"]

        mu_0 = d["mu_0"]
        mu_1 = d["mu_1"]
        mu_2 = d["mu_2"]

        # soil albedo conditions; ref eq. 8b
        I_down = 2 * np.pi * (mu_1 * yb[1] + mu_2 * yb[0])  # irradiance at bottom of canopy
        R_reflect = (
            rho / np.pi * (I_down + direct * mu_0 * np.pi * R_at_top * np.exp(-G * LAI / mu_0))
        )

        # bcs
        #    return np.vstack((ya[0] - I2d_top,    # down beams at top of canopy
        #                      ya[1] - I1d_top,
        #                      yb[2] - I_reflect,  # up beams at soil surface (x = LAI)
        #                      yb[3] - I_reflect))
        return np.array([ya[0] - R2d_top, ya[1] - R1d_top, yb[2] - R_reflect, yb[3] - R_reflect])

    # ----------------------------------------------------------------------------------------------
    # Begin implementation

    K = K_b  # for black leaves; should grey leaf version, K_b * k_prime, be used ???

    LAI = lai[0]  # total LAI
    P = 1  # probably phase function ???, 1 indicates isotropic scattering <-- note: not exactly satisfied in the leaf optical data (refl doesn't always = transmit)
    G = G_fn(psi)
    G_int_1 = integrate.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), 0, mu_s)[0]
    G_int_2 = integrate.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), mu_s, 1)[0]
    # ^ note: Barr code did not do these integrals, assumed a constant G fn

    # > allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all = np.zeros(s)

    for i in range(nbands):  # run for each band individually
        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i]  # W / m^2
        I_df0 = I_df0_all[i]

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
        r = leaf_r[i]  # leaf element reflectance
        t = leaf_t[i]  # leaf element transmittance
        omega = r + t  # single-scattering albedo

        # ------------------------------------------------------
        # calculate params; ref eqs. 4
        # note: assuming constant P for now, so many params have the same value
        #   should move the integration outside loop and store the value since it is the same

        mu_1 = 0.5 * mu_s ** 2
        mu_2 = 0.5 * (1 - mu_s ** 2)
        alpha_p = 0.5 * omega * P * (1 - mu_s) * G_int_2
        alpha_m = alpha_p
        beta_p = 0.5 * omega * P * (1 - mu_s) * G_int_1
        beta_m = beta_p
        gamma_p = 0.5 * omega * P * (mu_s - 0) * G_int_1
        gamma_m = gamma_p
        eps_p1 = 0.25 * omega * R_dr0 * P * (mu_s - 0)
        eps_m1 = eps_p1
        eps_p2 = 0.25 * omega * R_dr0 * P * (1 - mu_s)
        eps_m2 = eps_p2
        k_p1 = G_int_1
        k_m1 = G_int_1  # note Barr code did not have negative here. This seems to be right thing to do. at least needs abs()
        k_p2 = G_int_2
        k_m2 = G_int_2

        # create dict for input into the fns for the BVP solultion process
        #   probably not currently necessary since these vars are global within the model fn
        #   but would be helpful if redesign
        d = {
            "G": G,
            "LAI": LAI,
            "rho": rho,
            "mu_0": mu,
            "mu_1": mu_1,
            "mu_2": mu_2,
            "mu_s": mu_s,
            "alpha_p": alpha_p,
            "alpha_m": alpha_m,
            "beta_p": beta_p,
            "beta_m": beta_m,
            "gamma_p": gamma_p,
            "gamma_m": gamma_m,
            "eps_p1": eps_p1,
            "eps_m1": eps_m1,
            "eps_p2": eps_p2,
            "eps_m2": eps_m2,
            "k_p1": k_p1,
            "k_m1": k_m1,
            "k_p2": k_p2,
            "k_m2": k_m2,
        }

        # ------------------------------------------------------
        # contribution to diffuse due to scattering of direct beam

        x = np.linspace(0, LAI, 50)
        y0 = np.ones((4, x.size))  # initial guess

        fun = lambda x, y: eqns(x, y, d, direct=1)  # noqa: E731
        bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=1, R0=R_dr0)  # noqa: E731

        res = integrate.solve_bvp(fun, bcs, x, y0, tol=1e-6)

        y = res.sol(lai)  # solution splined over LAI vals

        # compute irradiances for the 4 streams from the radiance ("intensity") solution
        I2d_dr = 2 * np.pi * mu_2 * y[0, :]
        I1d_dr = 2 * np.pi * mu_1 * y[1, :]
        I1u_dr = 2 * np.pi * mu_1 * y[2, :]
        I2u_dr = 2 * np.pi * mu_2 * y[3, :]

        # ------------------------------------------------------
        # contribution to diffuse due to attenuation of canopy-incident diffuse

        #    x = np.linspace(0, LAI, 50)
        #    y0 = np.ones((4, x.size))  # initial guess

        fun = lambda x, y: eqns(x, y, d, direct=0)  # noqa: E731
        bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=0, R0=R_df0)  # noqa: E731

        res = integrate.solve_bvp(fun, bcs, x, y0, tol=1e-6)

        y = res.sol(lai)  # solution splined over LAI vals

        # compute irradiances for the 4 streams from the radiance ("intensity") solution
        # each stream has a brother region on the -mu side of the hemisphere => x2
        I2d_df = 2 * np.pi * mu_2 * y[0, :]
        I1d_df = 2 * np.pi * mu_1 * y[1, :]
        I1u_df = 2 * np.pi * mu_1 * y[2, :]
        I2u_df = 2 * np.pi * mu_2 * y[3, :]

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
        I_dr_all[:, i] = I_dr
        I_df_d_all[:, i] = I_df_d
        I_df_u_all[:, i] = I_df_u
        F_all[:, i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d

    # return I_dr_all, I_df_d_all, I_df_u_all, F_all
    return dict(I_dr=I_dr_all, I_df_d=I_df_d_all, I_df_u=I_df_u_all, F=F_all)
