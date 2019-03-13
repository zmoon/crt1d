"""
Parameterizations of the impact of leaf angles on canopy RT.

"""

import numpy as np

def G_ellipsoidal(psi, x):
    """Leaf angle factor ("big G") for the ellipsoidal leaf angle distribution.

    ref: Campbell (1986) eqs. 5, 6

    the ROB@UVa code set x = 0.96 instead of using the calculated 'orient' value ???

    Inputs
    ------
    psi: zenith angle in radians
    x:   b/a, ratio of horizontal semixaxis length to vertical, s.t. x > 0 indicates oblate spheroid

    """

    phi = np.pi / 2 - psi  # elevation angle

    p1 = np.sqrt( x**2 + 1/(np.tan(phi)**2) )  # numerator
    if x > 1:
        eps1 = np.sqrt(1 - x**-2)
        p2 = x + 0.5 * eps1 * x * np.log( (1 + eps1)*(1 - eps1) )  # denom

    else:
        eps2 = np.sqrt(1 - x**2)
        p2 = x + np.arcsin(eps2) / eps2  # denom

    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)




