# fmt: off
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize


short_name = '2s'
long_name = 'Dickinson–Sellers two-stream'

def solve_2s(*, psi,
    I_dr0_all, I_df0_all,
    lai,
    leaf_t, leaf_r, soil_r,
    K_b_fn, G_fn,
    mla,
    ):
    """Dickinson-Sellers 2-stream solution,
    the most common scheme used in regional/climate models.

    Implementation follows Dickinson (1983) and Sellers (1985)
    (mainly Sellers -- variable names chosen to match his)
    and includes the minor correction from the later Sellers paper (1996).

    """
    K_b = K_b_fn(psi)
    mu = np.cos(psi)
    theta_bar = np.deg2rad(mla)  # mean leaf inclination angle, deg->rad; eq. 3

    # mu_bar := average inverse diffuse optical depth, per unit leaf area; p. 1336
    #   sa: angle of scattered flux
    mu_bar  = integrate.quad(lambda sa: np.cos(sa) / G_fn(sa) * -np.sin(sa), np.pi/2, 0)[0]  # p. 1336
    mu_bar2 = integrate.quad(lambda mu_prime: mu_prime / G_fn(np.arccos(mu_prime)), 0, 1)[0]
    # TODO: following could be another optional check
    # assert( mu_bar == mu_bar2 )
    assert( np.isclose(mu_bar, mu_bar2) )

    L_T = lai[0]  # total LAI

    K = K_b  # for black leaves; should grey leaf version, K_b * k_prime, be used ???

    #> allocate arrays in which to save the solutions for each band
    nbands = I_dr0_all.size
    nz = lai.size
    s = (nz, nbands)  # to make pylint shut up until it supports _like()
    I_dr_all   = np.zeros(s)
    I_df_d_all = np.zeros(s)
    I_df_u_all = np.zeros(s)
    F_all      = np.zeros(s)

    #
    #> run for each band individually
    #
    for i in range(nbands):

        # calculate top-of-canopy irradiance present in the band
        I_dr0 = I_dr0_all[i]  # W / m^2
        I_df0 = I_df0_all[i]

        # load canopy optical properties
        alpha = leaf_r[i]  # leaf element reflectance
        tau   = leaf_t[i]  # leaf element transmittance
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
        # I know I wouldn't have done this if there wasn't one (an issue)
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

        # TODO: make checks like this optional for solvers with a `check` kwarg
        # for name, arr in {'I_df_u_dr': I_df_u_dr, 'I_df_u_df': I_df_u_df, 'I_df_d_dr': I_df_d_dr, 'I_df_d_df': I_df_d_df}.items():
        #     # if np.any(arr < -np.finfo(float).eps):
        #     if np.any(arr < -1e-12):
        #         print(f"{name} has elements < 0")
        #         print(arr)

        # Beer--Lambert direct beam attenuation
        I_dr = I_dr0 * np.exp(-K * lai)


        # save
        I_dr_all[:,i] = I_dr
        I_df_d_all[:,i] = I_df_d
        I_df_u_all[:,i] = I_df_u
        F_all[:,i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d


    # return I_dr_all, I_df_d_all, I_df_u_all, F_all
    return dict(\
        I_dr = I_dr_all,
        I_df_d = I_df_d_all,
        I_df_u = I_df_u_all,
        F = F_all
        )
