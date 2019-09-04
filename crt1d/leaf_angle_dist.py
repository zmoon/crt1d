"""
Parameterizations of the impact of leaf angles on canopy RT.

Leaf angle factor G and the extinction coeff K have the following relationship
    K = G / cos(psi)

"""

import numpy as np

def G_ellipsoidal(psi, x):
    """G for the ellipsoidal leaf angle distribution.

    ref: Campbell (1986) eqs. 5, 6

    Inputs
    ------
    psi: zenith angle in radians
    x:   b/a, ratio of horizontal semixaxis length to vertical, s.t. x > 1 indicates oblate spheroid

    """

    phi = np.pi/2 - psi  # elevation angle

    p1 = np.sqrt( x**2 + 1/(np.tan(phi)**2) )  # numerator
    if x > 1:  # something still wrong with this formula!. G should increase with x
        eps1 = np.sqrt(1 - x**-2)
        p2 = x + 0.5*eps1*x*np.log( (1 + eps1)/(1 - eps1) )  # denom

    else:
        eps2 = np.sqrt(1 - x**2)
        p2 = x + np.arcsin(eps2)/eps2  # denom

    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)


def G_ellipsoidal_approx(psi, x):
    """Campbell G approximate form.

    References
    ----------
    area ratio term: Campbell (1990) eq. 14
    exact formula:   Cambell & Norman (1996) eq. 15.4

    """

    p1 = np.sqrt(x**2 + np.tan(psi)**2)
    p2 = x + 1.774*(x+1.182)**-0.733
    K = p1/p2

    return K * np.cos(psi)  # K = G / cos(psi)


