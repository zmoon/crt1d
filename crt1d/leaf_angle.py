"""
Parameterizations of the impact of leaf angles on canopy RT.

Leaf angle factor :math:`G` and the black leaf extinction coeff :math:`K_b`
have the following relationship:

.. math::
   K_b = G / \cos(\psi)

where :math:`\psi` is the solar zenith angle.

:math:`G` is the mean projection of leaf area in the direction psi.
"""
import numpy as np
from scipy import integrate

PI = np.pi

# note that Bonan uses g for azimuth angle dist and f for inclination angle dist
# here we neglect any azimuth angle preference


def g_spherical(theta_l):
    """PDF of leaf inclination angle for spherical distribution.
    Vertical leaves are favored, but not as much so as for erectophile.
    """
    return np.sin(theta_l)


def g_uniform(theta_l):
    return 2 / PI  # note no theta_l dependence


def g_planophile(theta_l):
    """Mostly horizontal."""
    return 2 / PI * (1 + np.cos(2 * theta_l))


def g_erectophile(theta_l):
    """Mostly vertical."""
    return 2 / PI * (1 - np.cos(2 * theta_l))


def g_plagiophile(theta_l):
    """Between horizontal and vertical."""
    return 2 / PI * (1 - np.cos(4 * theta_l))


def mla_from_g(g_fn):
    r"""Calculate (estimate) the mean leaf inclination angle (deg.)
    by numerically integrating the distribution's PDF: g(psi).
    """
    fun = lambda x: x * g_fn(x)
    theta_l_bar = integrate.quad(fun, 0, PI / 2)[0]  # returns (y, err)
    return np.rad2deg(theta_l_bar)


def G_horizontal(psi):
    """G for horizontal leaves."""
    return np.cos(psi)


def G_spherical(psi):
    """G for spherical leaf inclination angle distribution."""
    return 0.5  # note no psi dependence


def G_vertical(psi):
    """G for vertical leaves."""
    return 2 / PI * np.sin(psi)


def g_ellipsoidal(theta_l, x):
    """PDF of leaf inclination angle for ellipsoidal distribution.
    Following Bonan (2019) p. 30, eqs. 2.11-14
    """
    # note Campbell (1990) uses "Lambda" instead of Bonan's "l"
    if x < 1:
        e1 = np.sqrt(1 - x ** 2)
        l = x + np.arcsin(e1) / e1
    elif x == 1:  # => spherical
        l = 2
    else:  # x > 1
        e2 = np.sqrt(1 - x ** -2)
        l = x + np.log((1 + e2) / (1 - e2)) / (2 * e2 * x)

    # eq. 2.11 -- numerator and denominator
    p1 = 2 * x ** 3 * np.sin(theta_l)
    p2 = (np.cos(theta_l) ** 2 + x ** 2 * np.sin(theta_l) ** 2) ** 2

    return p1 / (l * p2)


def G_ellipsoidal(psi, x):
    """G for the ellipsoidal leaf angle distribution.

    ref: Campbell (1986) eqs. 5, 6 (:cite:`campbell_extinction_1986`)

    Parameters
    ----------
    psi : float
        zenith angle in radians
    x : float
        b/a, ratio of horizontal semixaxis length to vertical,
        s.t. x > 1 indicates oblate spheroid
    """
    if x == 1:  # => spherical
        res = np.full_like(psi, G_spherical(psi))  # allow psi array input
        return float(res) if res.size == 1 else res
        # TODO: maybe create helper fn for this issue

    phi = PI / 2 - psi  # elevation angle

    p1 = np.sqrt(x ** 2 + 1 / (np.tan(phi) ** 2))  # numerator
    if x > 1:
        eps1 = np.sqrt(1 - x ** -2)
        p2 = x + 1 / (2 * eps1 * x) * np.log((1 + eps1) / (1 - eps1))  # denom
        # ^ note: the paper says (1 / 2 eps1 x) but it should be 1/(2 eps1 x)
    else:
        eps2 = np.sqrt(1 - x ** 2)
        p2 = x + np.arcsin(eps2) / eps2  # denom

    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)


def G_ellipsoidal_approx(psi, x):
    """Campbell G approximate form.

    References
    ----------
    area ratio term: Campbell (1990) eq. 14
    exact formula: Campbell & Norman (1996) eq. 15.4

    """
    p1 = np.sqrt(x ** 2 + np.tan(psi) ** 2)
    p2 = x + 1.774 * (x + 1.182) ** -0.733
    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)


def x_to_mla_approx(x):
    r"""Convert x to mean leaf angle (deg.)
    for the ellipsoidal leaf angle distribution.
    Using Campbell (1990) eq. 16.
    """
    theta_l_bar = 9.65 * (3 + x) ** (-1.65)
    return np.rad2deg(theta_l_bar)


def x_to_mla_integ(x):
    """Convert x to mean leaf angle (deg.)
    for the ellipsoidal leaf angle distribution
    by numerically integrating the leaf angle PDF.
    """
    return mla_from_g(lambda psi: g_ellipsoidal(psi, x))


def mla_to_x_approx(mla):
    r"""Convert mean leaf angle (deg.) to x
    for the ellipsoidal leaf angle distribution.
    Using Campbell (1990) eq. 16 inverted.
    """
    x = (np.deg2rad(mla) / 9.65) ** (-1.0 / 1.65) - 3.0
    assert x > 0
    return x


# TODO: better mla_to_x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    # TODO: leaf angle distribution PDFs

    # x to mla
    fig, ax = plt.subplots()
    ax.set_title("Ellipsoidal mean leaf angle from $x$")
    x = np.linspace(1, 10, 200)
    ax.plot(x, [x_to_mla_integ(xi) for xi in x], label="numerical integration of exact PDF")
    ax.plot(x, x_to_mla_approx(x), label="approximation")
    ax.set(xlabel="$x$", ylabel="mean leaf angle (deg.)")
    ax.legend()
    fig.tight_layout()

    # G ellipsoidal exact formulation vs approx
    fig, ax = plt.subplots()
    ax.set_title("Ellipsoidal $G$")
    sza = np.linspace(0, 85, 200)
    psi = np.deg2rad(sza)
    for xval in [0.5, 1, 2, 4]:
        ax.plot(sza, G_ellipsoidal(psi, xval), label=f"analytical, $x={xval}$")
        ax.plot(sza, G_ellipsoidal_approx(psi, xval), label=f"approx., $x={xval}$")
    ax.set(xlabel="solar zenith angle (deg.)", ylabel="$G$")
    ax.legend()
    fig.tight_layout()
