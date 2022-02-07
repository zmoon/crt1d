r"""
Parameterizations of the impact of leaf angles on canopy RT.

Leaf angle factor :math:`G` and the black leaf extinction coeff :math:`K_b`
have the following relationship:

.. math::
   K_b = G / \cos(\psi)

where :math:`\psi` is the solar zenith angle and :math:`K_b = K_b(\psi), G = G(\psi)`.

:math:`G` is the mean relative projection of leaf area in the direction :math:`\psi`.

:math:`g(\theta_l)` is the PDF of leaf inclination angle :math:`\theta_l`
(relative to the horizontal plane).
:math:`G(\psi)` functions are derived from these distributions.
The azimuth angle is usually assumed to have a uniform distribution
and so does not have an impact.
"""
import numpy as np
from scipy import integrate
from scipy import optimize

PI = np.pi

# note that Bonan uses $g$ for azimuth angle dist and $f$ for inclination angle dist
# here we neglect any azimuth angle preference


def g_spherical(theta_l):
    r"""PDF of :math:`\theta_l` for the spherical distribution.
    Vertical leaves are favored, but not as much so as for erectophile.
    """
    return np.sin(theta_l)


def g_uniform(theta_l):
    r"""PDF of :math:`\theta_l` for a uniform distribution."""
    # note no `theta_l` dependence
    val = 2 / PI
    return val if np.isscalar(theta_l) else np.full_like(theta_l, val)


def g_planophile(theta_l):
    r"""PDF of :math:`\theta_l` for a mostly horizontal distribution."""
    return 2 / PI * (1 + np.cos(2 * theta_l))


def g_erectophile(theta_l):
    r"""PDF of :math:`\theta_l` for a mostly vertical distribution."""
    return 2 / PI * (1 - np.cos(2 * theta_l))


def g_plagiophile(theta_l):
    r"""PDF of :math:`\theta_l` for a distribution between horizontal and vertical."""
    return 2 / PI * (1 - np.cos(4 * theta_l))


def g_ellipsoidal(theta_l, x):
    r"""PDF of :math:`\theta_l` for the ellipsoidal distribution
    with parameter `x`.
    Following :cite:t:`bonan_climate_2019` (p. 30, eqs. 2.11--14).
    """
    # note Campbell (1990) uses "Λ" (Lambda) instead of Bonan's "l"
    if x < 1:
        e1 = np.sqrt(1 - x**2)
        l = x + np.arcsin(e1) / e1  # noqa: E741 ambiguous name
    elif x == 1:  # => spherical
        l = 2  # noqa: E741
    else:  # x > 1
        e2 = np.sqrt(1 - x**-2)
        l = x + np.log((1 + e2) / (1 - e2)) / (2 * e2 * x)  # noqa: E741

    # eq. 2.11 -- numerator and denominator
    p1 = 2 * x**3 * np.sin(theta_l)
    p2 = (np.cos(theta_l) ** 2 + x**2 * np.sin(theta_l) ** 2) ** 2

    return p1 / (l * p2)


def mla_from_g(g_fn):
    r"""Calculate (estimate) the mean leaf inclination angle (deg.)
    by numerically integrating the distribution's PDF: :math:`g(\psi)`.
    """
    theta_l_bar = integrate.quad(lambda x: x * g_fn(x), 0, PI / 2)[0]  # returns (y, err)
    return np.rad2deg(theta_l_bar)


def xl_from_g(g_fn):
    r"""Compute :math:`\chi_l`, an index which quantifies the departure of the
    leaf angle distribution from spherical.

    Vertical leaves have :math:`\chi_l = -1` and horizontal leaves :math:`\chi_l = +1`.

    :cite:t:`bonan_climate_2019` eq. 2.16
    """
    xl = (
        0.5
        * integrate.quad(
            lambda theta_l: np.abs(np.sin(theta_l) - g_fn(theta_l)),
            0,
            PI / 2,
        )[0]
    )

    # Per Bonan, sign should be determined using the [60, 90] region
    F3 = integrate.quad(
        lambda theta_l: np.sin(theta_l) - g_fn(theta_l),
        PI / 3,
        PI / 2,
    )[0]
    sign = np.sign(F3)

    return xl * sign


def G_horizontal(psi):
    r""":math:`G(\psi)` for horizontal leaves."""
    return np.cos(psi)


def G_spherical(psi):
    r""":math:`G(\psi)` for the spherical leaf inclination angle distribution."""
    return 0.5  # note no `psi` dependence


def G_vertical(psi):
    r""":math:`G(\psi)` for vertical leaves."""
    return 2 / PI * np.sin(psi)


def G_ellipsoidal(psi, x):
    r""":math:`G(\psi)` for the ellipsoidal leaf angle distribution
    with parameter `x`.

    *Reference*: Campbell (1986) eqs. 5, 6 :cite:`campbell_extinction_1986`

    Parameters
    ----------
    psi : float
        Solar zenith angle in radians.
    x : float
        b/a -- the ratio of ellipse horizontal semixaxis length to vertical,
        s.t. `x` > 1 indicates an oblate spheroid.
    """
    if x == 1:  # => spherical
        res = np.full_like(psi, G_spherical(psi))  # allow psi array input
        return float(res) if res.size == 1 else res
        # TODO: maybe create helper fn for this issue

    phi = PI / 2 - psi  # elevation angle

    p1 = np.sqrt(x**2 + 1 / (np.tan(phi) ** 2))  # numerator
    if x > 1:
        eps1 = np.sqrt(1 - x**-2)
        p2 = x + 1 / (2 * eps1 * x) * np.log((1 + eps1) / (1 - eps1))  # denom
        # ^ note: the paper says (1 / 2 eps1 x) but it should be 1/(2 eps1 x)
    else:
        eps2 = np.sqrt(1 - x**2)
        p2 = x + np.arcsin(eps2) / eps2  # denom

    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)


def G_ellipsoidal_approx(psi, x):
    """Campbell :math:`G` approximate form.

    References
    ----------
    * area ratio term: Campbell (1990) eq. 14 :cite:`campbellDerivationAngleDensity1990`
    * exact formula: Campbell & Norman eq. 15.4 :cite:`campbell_introduction_2012`
    """
    p1 = np.sqrt(x**2 + np.tan(psi) ** 2)
    p2 = x + 1.774 * (x + 1.182) ** -0.733
    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)


def G_ellipsoidal_approx_bonan(psi, xl):
    r"""Campbell :math:`G` approximate form -- Bonan version.

    This uses :math:`\chi_l`, an index which quantifies the departure of the
    leaf angle distribution from spherical.

    .. warning::
       `xl` is not the same parameter as the ``x`` used elsewhere in this module.
       ``xl=0`` gives spherical, whereas ``x=1`` gives spherical.
    """
    # TODO: add converter fn from `xl` to `x`?

    # Clip $\chi_l$ to [-0.4, 0.6]
    chil = min(max(xl, -0.4), 0.6)

    # Ross-Goudriaan function terms
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil**2
    phi2 = 0.877 * (1 - 2 * phi1)

    return phi1 + phi2 * np.cos(psi)


def x_to_mla_approx(x):
    r"""Convert `x` to mean leaf angle (deg.)
    for the ellipsoidal leaf angle distribution.
    Using Campbell (1990) :cite:`campbellDerivationAngleDensity1990` eq. 16.
    """
    theta_l_bar = 9.65 * (3 + x) ** (-1.65)
    return np.rad2deg(theta_l_bar)


def x_to_mla_integ(x):
    """Convert `x` to mean leaf angle (deg.)
    for the ellipsoidal leaf angle distribution
    by numerically integrating the leaf angle PDF.
    """
    return mla_from_g(lambda psi: g_ellipsoidal(psi, x))


def mla_to_x_approx(mla):
    r"""Convert mean leaf angle (deg.) to `x`
    for the ellipsoidal leaf angle distribution.
    Using Campbell (1990) :cite:`campbellDerivationAngleDensity1990` eq. 16 inverted.
    """
    x = (np.deg2rad(mla) / 9.65) ** (-1.0 / 1.65) - 3.0
    assert x > 0
    return x


def mla_to_x_integ(mla):
    r"""Convert mean leaf angle (deg.) to `x`
    for the ellipsoidal leaf angle distribution
    by optimization.
    """
    res = optimize.minimize_scalar(
        lambda x: np.abs(x_to_mla_integ(x) - mla),
        bounds=(0, 999),  # note arbitrary `x` upper bound (setting `None` doesn't work)
        method="bounded",
        options=dict(
            xatol=1e-8,  # default: 1e-5
        ),
    )
    return res.x
