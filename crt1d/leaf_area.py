"""
Distribute leaf area amongst model layers.

Based on a quantitative description of the leaf area profile,
and desired number of model layers,
allocate equal increments of dlai to layers,
and assign heights (z; m) to give the desired leaf area profile.

LAI is cumulative and defined at the model interface levels.
"""
from collections import namedtuple

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.interpolate import interp1d
from scipy.stats import beta
from scipy.stats import gamma

# from scipy.special import beta as beta_fn
# TODO: move to standard scipy imports

__all__ = [
    "distribute_lai_beta",
    "distribute_lai_beta_bonan",
    "distribute_lai_weibull_z",
    "distribute_lai_weibull",
    "distribute_lai_gamma",
    "distribute_lai_from_cdd",
]


_LeafAreaProfile = namedtuple("LeafAreaProfile", "lai lad z")


def distribute_lai_beta(h_c, LAI, n, *, h_min=0.5):
    """Create LAI profile using beta distribution for leaf area density.
    This uses a set fractional height of maximum LAD and corresponding beta distribution shape.

    Parameters
    ----------
    h_c : float
        canopy height
    n : int
        number of layers (LAI/interface levels)
    LAI : float
        total LAI
    h_min : float
        height above ground of the canopy bottom

    Returns
    -------
    NamedTuple
        lai : numpy.ndarray
            cumulative LAI profile; accumulates from top, so decreasing with z
        lad : numpy.ndarray
            leaf area density profile
        z : numpy.ndarray
            height above ground (increasing)

    """
    h_max_lad = 0.7 * h_c  # TODO: make this an input
    d_max_lad = h_c - h_max_lad

    # mode is (a-1)/(a+b-2)
    d = d_max_lad / h_c  # relative depth = desired mode
    b = 3
    a = -((b - 2) * d + 1) / (d - 1)

    # lad = np.zeros((n, ))
    lai = np.zeros((n,))
    z = np.zeros((n,))

    dist = beta(a, b)

    lai_cum_pct = np.linspace(1.0, 0, n)
    # beta pdf is confined to [0, 1] unlike Gamma

    z = (h_c - h_min) * (1 - dist.ppf(lai_cum_pct)) + h_min
    # TODO: fix the h_min stuff to be consistent with h_max_lad

    lai = lai_cum_pct * LAI

    zrel = (z - h_min) / (h_c - h_min)
    lad = LAI / (h_c - h_min) * beta.pdf(zrel, b, a)

    return _LeafAreaProfile(lai, lad, z)


def test_plot_distribute_lai_beta():
    res = distribute_lai_beta(10, 5, 20)
    test_plot_distribute_lai_res(res, title="distribute_lai_beta")


def distribute_lai_beta_bonan(h_c, LAI, n, *, h_min=0.5, p=3.5, q=2.0):
    """Beta distribution based on Bonan SP 2.1.

    Parameters
    ----------
    h_c : float
        canopy height
    n : int
        number of layers (LAI/interface levels)
    LAI : float
        total LAI
    h_min : float
        height above ground of the canopy bottom
    p, q : float
        beta distribution shape parameters

    Returns
    -------
    NamedTuple
        lai : numpy.ndarray
            cumulative LAI profile; accumulates from top, so decreasing with z
        lad : numpy.ndarray
            leaf area density profile
        z : numpy.ndarray
            height above ground (increasing)
    """
    dist = beta(p, q)

    # depth of canopy
    h = h_c - h_min

    # fractional cumulative LAI is 1 at z=0 and 0 at the canopy top
    lai_cum_frac = np.linspace(1.0, 0, n)  # increasing from top->bottom; decreasing with z

    # z is increasing
    z = (h * dist.ppf(lai_cum_frac) + h_min)[::-1]

    lai = lai_cum_frac * LAI

    # relative height in canopy (not necessarily relative height above ground z/h_c)
    zrel = (z - h_min) / h

    # lad = LAI/h * (zrel**(p-1) * (1-zrel)**(q-1)) / beta_fn(q, p)  # equivalent
    lad = LAI / h * dist.pdf(zrel)

    return _LeafAreaProfile(lai, lad, z)


def test_plot_distribute_lai_beta_bonan():
    res = distribute_lai_beta_bonan(10, 5, 20)
    test_plot_distribute_lai_res(res, title="distribute_lai_beta_bonan")


# adapted from: https://github.com/LukeEcomod/pyAPES_skeleton/blob/master/tools/utilities.py
# unlike the above, z is a required input
def distribute_lai_weibull_z(z, LAI, h, hb=0.0, *, b=None, c=None, species=None):
    """Leaf area profile from Weibull distribution for given `z` grid.

    Samuli's notes:

    .. code:: none

       Generates leaf-area density profile from Weibull-distribution
       Args:
           z: height array (m), monotonic and constant steps
           LAI: leaf-area index (m2m-2)
           h: canopy height (m), scalar
           hb: crown base height (m), scalar
           b: Weibull shape parameter 1, scalar
           c: Weibull shape parameter 2, scalar
           species: 'pine', 'spruce', 'birch' to use table values
       Returns:
           LAD: leaf-area density (m2m-3), array
       SOURCE:
           Teske, M.E., and H.W. Thistle, 2004, A library of forest canopy structure for
           use in interception modeling. Forest Ecology and Management, 198, 341-350.
           Note: their formula is missing brackets for the scale param.
           Here their profiles are used between hb and h
       AUTHOR:
           Gabriel Katul, 2009. Coverted to Python 16.4.2014 / Samuli Launiainen
    """

    para = {"pine": [0.906, 2.145], "spruce": [2.375, 1.289], "birch": [0.557, 1.914]}

    if (max(z) <= h) | (h <= hb):
        raise ValueError("h must be lower than uppermost gridpoint")

    if b is None or c is None:
        b, c = para[species]

    z = np.array(z)
    dz = abs(z[1] - z[0])
    N = np.size(z)
    LAD = np.zeros(N)

    a = np.zeros(N)

    # dummy variables
    ix = np.where((z > hb) & (z <= h))[0]
    x = np.linspace(0, 1, len(ix))  # normalized within-crown height

    # weibul-distribution within crown
    cc = (
        -(c / b)
        * (((1.0 - x) / b) ** (c - 1.0))
        * (np.exp(-(((1.0 - x) / b) ** c)))
        / (1.0 - np.exp(-((1.0 / b) ** c)))
    )

    a[ix] = cc
    a = np.abs(a / sum(a * dz))

    LAD = LAI * a

    # numerically estimate the cumulative LAI profile from the LAD profile
    lai = -1 * integrate.cumtrapz(LAD[::-1], z[::-1], initial=0)[::-1]

    return _LeafAreaProfile(lai, LAD, z)


def test_plot_distribute_lai_weibull_z():
    z = np.linspace(0, 10.5, 20)

    for spc in ["birch", "pine", "spruce"]:
        res = distribute_lai_weibull_z(z, LAI=5, h=10, hb=2, species=spc)
        test_plot_distribute_lai_res(res, title=f"distribute_lai_weibul_z {spc}")


def distribute_lai_weibull(h_c, LAI, n, *, h_min=0.5, b=None, c=None, species=None):
    """Leaf area profile with equal LAI increments from the Weibull distribution
    using :func:`distribute_lai_weibull_z` and interpolation (as such, lad(z) and
    lai(z) might not be completely consistent).

    Parameters
    ----------
    h_c : float
        canopy height
    n : int
        number of layers (LAI/interface levels)
    LAI : float
        total LAI
    h_min : float, optional
        Canopy minimum height above ground. The default is 0.5.
    b : float, optional
        Weibull shape parameter 1. The default is None.
    c : TYPE, optional
        Weibull shape parameter 2. The default is None.
    species : str, optional
        One of 'spruce', 'spruce', or 'birch'. Used to look up typical shape parameter values.
        The default is None.

    Returns
    -------
    NamedTuple
        lai : numpy.ndarray
            cumulative LAI profile; accumulates from top, so decreasing with z
        lad : numpy.ndarray
            leaf area density profile
        z : numpy.ndarray
            height above ground (increasing)
    """
    z0 = np.linspace(0, h_c + 0.5, n)

    res0 = distribute_lai_weibull_z(z0, LAI, h_c, hb=h_min, b=b, c=c, species=species)

    # we use the evenly spaced z value LAI profile to find another with evenly spaced LAI
    lai = np.linspace(LAI, 0, n)
    fi_lai = interp1d(res0.lai[z0 >= h_min], z0[z0 >= h_min], bounds_error=False)
    # the dist is just cut to zero below h_min, so don't want to use that part
    z = np.r_[h_min, fi_lai(lai[1:])]

    # use the new z values to get interpolated lad
    fi_lad = interp1d(z0, res0.lad)
    lad = np.r_[0, fi_lad(z[1:])]

    # TODO: doesn't work perfectly. would be better to use the Weibull distribution
    # from scipy.stats and define this one similar to the beta ones

    # correction factor
    c = LAI / lai[0]
    lai *= c
    lad *= c

    return _LeafAreaProfile(lai, lad, z)


def test_plot_distribute_lai_weibull():
    for spc in ["birch", "pine", "spruce"]:
        res = distribute_lai_weibull(h_c=10, LAI=5, n=20, h_min=2, species=spc)
        test_plot_distribute_lai_res(res, title=f"distribute_lai_weibul {spc}")


def distribute_lai_gamma(h_c, LAI, n):
    """Create LAI profile using Gamma distribution for leaf area density.

    Parameters
    ----------
    h_c : float
        canopy height
    n : int
        number of layers (LAI/interface levels)
    LAI : float
        total LAI

    Returns
    -------
    NamedTuple
        lai : numpy.ndarray
            cumulative LAI profile; accumulates from top, so decreasing with z
        lad : numpy.ndarray
            leaf area density profile
        z : numpy.ndarray
            height above ground (increasing)
    """
    h_max_lad = 0.7 * h_c  # TODO: could be a param
    d_max_lad = h_c - h_max_lad  # canopy depth

    b = 3.5
    a = d_max_lad / b + 1

    # a = 2.5
    # b = d_max_lad/(a-1)

    g = gamma(a, b)

    # lad = np.zeros((n, ))
    lai = np.zeros((n,))
    z = np.zeros((n,))

    lai_cum_pct = np.linspace(0.99, 0.1, n - 2)

    z[-1] = h_c
    z[1:-1] = h_c - g.ppf(lai_cum_pct)  # Gamma ppf depends on depth into the canopy
    z[0] = 0

    lai[-1] = 0
    lai[1:-1] = lai_cum_pct * LAI
    lai[0] = LAI

    # TODO: derive and output LAD as well

    return _LeafAreaProfile(lai, None, z)


def test_plot_distribute_lai_gamma():
    res = distribute_lai_beta(10, 5, 20)
    test_plot_distribute_lai_res(res, title="distribute_lai_gamma")


def test_plot_distribute_lai_res(res, *, title=None):
    """For any LAI profile results, plot the cumulative LAI profile, analytical LAD profile at
    the LAI (interface) levels if provided, and dlai/dz at layer midpoints.
    This provides a visual check for consistency between the lai and lad profile formulations.
    """
    lai = res.lai
    dlai = lai[:-1] - lai[1:]

    lad = res.lad

    z = res.z  # increasing
    dz = np.diff(z)
    # assert np.all(dz > 0)  # assert increasing

    zm = z[:-1] + 0.5 * dz  # midpoints

    # assert np.isclose(np.sum(dlai), lai[0])

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)

    ax1.set_title("LAD")
    if lad is not None:
        ax1.plot(res.lad, z, ".-", label=f"from dist\ntrapz = {integrate.trapz(lad, z):.4g}")
        # note: the trapz converges to true total LAI as n increases (number of layers)
    ax1.plot(dlai / dz, zm, "s-", mfc="none", label="dlai/dz at midpts")
    ax1.legend()
    ax1.set_ylabel("$z$")

    ax2.set_title("LAI")
    ax2.plot(lai, z, ".-")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()


# TODO: multi-stories with the above dist fns. could rework the layer machinery below to do this


# --------------------------------------------------------------------------------------------------
# The below code was used to construct the Borden LAD profile and shouldn't generally
# be used otherwise (although could).


class layer:
    """Leaf area distribution between two points (local minima in the overall LAI profile)
    For example,
    a layer with pdf (normalized to fLAI) and corresponding cdf methods.

    The upper and lower parts of the layer have different functional forms.
    """

    def __init__(self, h1, lad_h1, hmax, LAI, h2, lad_h2):
        self.h1 = h1
        self.lad_h1 = lad_h1
        self.hmax = hmax
        self.LAI = LAI
        self.h2 = h2
        self.lad_h2 = lad_h2

        fun = (
            lambda X: integrate.quad(lambda h: self.pdf0(h, lai_mult=X), h1, h2)[0] - LAI
        )  # noqa: E731
        self.lai_mult = optimize.fsolve(fun, 0.1)[0]
        if self.lai_mult <= 0:
            print("desired LAI too small")

    def f_lw(self, h, lai_mult=1):
        """ """
        #        linear = lambda h: (h - self.h1) * 0.07 + self.lad_h1
        #        curve = lambda h: np.sqrt((h - self.h1)**2 - 1)
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))  # noqa: E731

        mult = 1 / (curve(self.hmax) + self.lad_h1) * lai_mult
        return curve(h) * mult + self.lad_h1

    def F_lw(self, h):
        """
        cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+sin(1.3+*+(x+-+a)+%2F+(b+-+a))+*+c+%2B+d
        """
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))  # noqa: E731
        a = self.h1
        b = self.hmax
        c = 1 / (curve(self.hmax) + self.lad_h1) * self.lai_mult
        d = self.lad_h1
        return 1 / 1.3 * c * (a - b) * np.cos(1.3 * (a - h) / (a - b)) + d * h + 0

    def f_up(self, h, lai_mult=1):
        """ """
        mult = self.f_lw(self.hmax, lai_mult) - self.lad_h2
        x = (h - self.hmax) / (self.h2 - self.hmax)
        return (-((x + 0.5) ** 2) + (x + 0.5) + 0.75) * mult + self.lad_h2

    def find_ub_up(self, lb, lai):
        """
        analytically find upper bound for integration that gives the desired lai
        with lower bound lb

        http://www.wolframalpha.com/input/?i=integrate+from+a+to+b++(-2*(x%2B0.5)%5E2+%2B+2*(x%2B0.5)+%2B+1.5)%2F2+%3D+c+solve+for+b
        """
        return NotImplementedError

    def F_up(self, h):
        """cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+(-(x+%2B+0.5)**2+%2B+(x+%2B+0.5)+%2B+0.75)+*+a+%2B+b
        """
        x = (h - self.hmax) / (self.h2 - self.hmax)
        dhdx = self.h2 - self.hmax
        a = self.f_lw(self.hmax, self.lai_mult) - self.lad_h2
        b = self.lad_h2
        return x * (-1.0 / 3 * a * x ** 2 + a + b) * dhdx

    def pdf0(self, h, lai_mult=1):
        """
        pdf before normalization to desired LAI
        LAD at hmax = 1
        """
        if h < self.hmax:  # lower portion
            return self.f_lw(h, lai_mult)
        else:  # upper portion
            return self.f_up(h, lai_mult)

    def pdf(self, h):
        """
        Using the lai multipler, integration now gives the desired LAI
        """
        if h >= self.h2:
            return np.nan
        if h >= self.hmax:  # upper portion
            return self.f_up(h, lai_mult=self.lai_mult)
        elif h >= self.h1:  # lower portion
            return self.f_lw(h, lai_mult=self.lai_mult)
        else:
            return np.nan

    def cdf(self, h):
        """ """
        lai_top = self.F_up(self.h2) - self.F_up(self.hmax)
        #        print 'lai_top', lai_top
        if h >= self.h2:
            return 0
        elif h >= self.hmax:  # if gets here, is also < self.h_top
            return self.F_up(self.h2) - self.F_up(h)
        elif h >= self.h1:  # if gets here, is also < self.h_max
            return (self.F_lw(self.hmax) - self.F_lw(h)) + lai_top
        else:
            return self.LAI


class canopy_lai_dist:
    """
    arbitrary number of nodes, for including emergent layer, lower story, upper story, etc.
    uses the layer() class to create layers

    """

    def __init__(self, h_bottom, layers, LAI):
        """
        h_bottom: point above ground where LAD starts increasing from 0
        LAI: total LAI of the canopy

        layers: list of dicts with the following keys (for if more than 1 layer)
            h_max: height of max LAD in the section
            h_top: height of top of layer
            lad_h_top: LAD at the point where the layer ends and the layer above starts
            fLAI: fraction of total LAI that is in this layer
        """
        self.h_bottom = h_bottom
        #        self.h_canopy = h_canopy
        self.LAItot = LAI

        self.h_tops = np.array([ld["h_top"] for ld in layers])

        h1 = h_bottom
        lad_h1 = 0
        for ld in layers:  # loop through the layer dicts (lds)

            hmax = ld["h_max"]
            LAI = ld["fLAI"] * self.LAItot
            h2 = ld["h_top"]
            lad_h2 = ld["lad_h_top"]

            # create instance of layer class
            l = layer(h1, lad_h1, hmax, LAI, h2, lad_h2)  # noqa: E741

            ld["pdf"] = l.pdf
            ld["cdf"] = l.cdf
            ld["LAI"] = LAI

            h1 = h2
            lad_h1 = lad_h2

        self.lds = layers
        self.LAIlayers = np.array([ld["LAI"] for ld in layers])

        # vectorize fns ?

    #        self.pdf = np.vectorize(self.pdf)  # this one doesn't work for some reason
    #        self.cdf = np.vectorize(self.cdf)

    def pdf(self, h):
        """
        note: could put code in here to support h being a list or np.ndarray,
        since np.vectorize seems to fail if an h value < h_bottom is included
        """
        # find relevant layer
        if h > self.h_tops.max() or h < self.h_bottom:
            return 0
        else:
            lnum = np.where(self.h_tops >= h)[0].min()
            #            print lnum
            return self.lds[lnum]["pdf"](h)

    def cdf(self, h):
        if h > self.h_tops.max():
            return 0
        else:
            lnum = np.where(self.h_tops >= h)[0].min()

            if lnum == len(self.lds) - 1:
                above = 0
            else:
                above = np.sum([self.LAIlayers[lnum + 1 :]])

            return self.lds[lnum]["cdf"](h) + above

    def inv_cdf(self, ub, lai):
        """
        find the lb that gives that desired lai from numerical integration of the pdf

        using scipy.optimize.fsolve
        """
        fun = lambda lb: integrate.quad(self.pdf, lb, ub)[0] - lai  # noqa: E731
        lb = optimize.fsolve(fun, ub - 1e-6)

        return lb


def distribute_lai_from_cdd(cdd, n):
    """Create LAI profile based on input descriptors of the leaf area distribution.

    Parameters
    ----------
    cdd : dict
        canopy description dict. contains many things. see the default CSV for example
    n : int
        desired number of levels (LAI/interface levels)
    """
    LAI = cdd["lai_tot"]  # total canopy LAI
    h_bottom = cdd["h_bot"][-1]  # canopy bottom is the bottom of the bottom-most layer

    # > form inputs to the layer class
    nlayers = len(cdd["lai_frac"])
    layers = []
    for i in range(nlayers):
        layer = dict(
            h_max=cdd["h_max_lad"][i],
            h_top=cdd["h_top"][i],
            lad_h_top=cdd["lad_h_top"][i],
            fLAI=cdd["lai_frac"][i],
        )
        layers.append(layer)

    cld = canopy_lai_dist(h_bottom, layers[::-1], LAI)  # form dist of leaf area

    #    h_canopy = cld.lds[-1]['h_top']  # canopy height is the top of the upper-most layer
    h_canopy = cdd["h_canopy"]

    dlai = float(LAI) / (n - 1)  # desired const lai increment
    #    print dlai

    lai = np.zeros((n,))
    z = h_canopy * np.ones_like(lai)  # bottoms of layers?

    ub = h_canopy
    LAIcum = 0
    for i in range(n - 2, -1, -1):  # build from top down

        if LAI - LAIcum < dlai:
            assert i == 0
            z[0] = h_bottom
            lai[0] = LAI

        else:
            lb = cld.inv_cdf(ub, dlai)
            z[i] = lb
            lai[i] = lai[i + 1] + dlai

            ub = lb
            LAIcum += dlai

    return _LeafAreaProfile(lai, None, z)


def test_plot_canopy_layer_class():
    # ----------------------------------------------------------------------------------------------
    # graphical tests of the layer class alone

    layer_test = layer(h1=2, lad_h1=0, hmax=4.5, LAI=1.3, h2=8, lad_h2=0.1)

    h = np.linspace(1, 9, 20)
    dh = np.diff(h)
    dh_step = dh[-1]
    midpts = h[:-1] + dh_step / 2

    lad0 = np.array([layer_test.pdf0(x) for x in h])
    lad = np.array([layer_test.pdf(x) for x in h])
    lai = np.array([layer_test.cdf(x) for x in h])
    dlai = np.diff(lai)
    lai2 = dh_step * np.nancumsum(lad[::-1])[::-1]
    lai3 = np.array([-integrate.quad(layer_test.pdf, 8, x)[0] for x in midpts])

    plt.figure()
    plt.title("Leaf area density")
    plt.plot(lad0, h, label="before LAI mult")
    plt.plot(lad, h, label="after")
    plt.plot(-dlai / dh, midpts, "m:", label="dlai at midpts")
    plt.xlim(xmin=0)
    plt.legend()

    plt.figure()
    plt.title("Leaf area index (cumulative)")
    plt.plot(lai, h, c="0.75", lw=4, label="lai using layer.cdf")
    plt.plot(lai2, h, "m:", label="lai by cumsum-ing layer.pdf results")
    plt.plot(lai3, midpts, "b--", label="lai by numerical integration of layer.pdf results")
    plt.xlim(xmin=0)
    plt.legend()


def test_plot_canopy_lai_dist():
    # ----------------------------------------------------------------------------------------------
    # graphical tests of full canopy_lai_dist

    l1 = {"h_max": 2.5, "h_top": 5, "lad_h_top": 0.1, "fLAI": 0.2}
    l2 = {"h_max": 8, "h_top": 13, "lad_h_top": 0.1, "fLAI": 0.35}
    l3 = {"h_max": 16, "h_top": 20, "lad_h_top": 0.05, "fLAI": 0.35}
    l4 = {"h_max": 21.5, "h_top": 23, "lad_h_top": 0, "fLAI": 0.1}

    cld1 = canopy_lai_dist(0.5, [l1, l2, l3, l4], 5)

    h = np.linspace(0, 24, 200)

    lad = np.array([cld1.pdf(x) for x in h])
    lai = np.array([cld1.cdf(x) for x in h])

    plt.figure()
    plt.plot(lad, h)
    plt.xlim(xmin=0)

    plt.figure()
    plt.plot(lai, h)
    plt.xlim(xmin=0)
    # ----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    # test_plot_distribute_lai_beta()
    # test_plot_distribute_lai_beta_bonan()
    test_plot_distribute_lai_weibull()
    test_plot_distribute_lai_weibull_z()
    # test_plot_distribute_lai_gamma()
    # test_plot_canopy_layer_class()
    # test_plot_canopy_lai_dist()
