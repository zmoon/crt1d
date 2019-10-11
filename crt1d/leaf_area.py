"""
The canopy_lai_dist class 

Based on a quantitative description of the leaf area profile,
and desired number of model layers,
allocate equal increments of dlai to layers,
and assign heights (z; m) to give the desired leaf area profile.

"""



import numpy as np
import scipy.integrate as si
#from scipy.interpolate import interp1d
import scipy.optimize as so
from scipy.stats import gamma, beta
# TODO: move to standard scipy imports




class layer:
    """Leaf area distribution between two points (local minima in the overall LAI profile)
    For example, 
    a layer with pdf (normalized to fLAI) and corresponding cdf methods
    """

    def __init__(self, h1, lad_h1, hmax, LAI, h2, lad_h2):
        self.h1 = h1
        self.lad_h1 = lad_h1
        self.hmax = hmax
        self.LAI = LAI
        self.h2 = h2
        self.lad_h2 = lad_h2

#        print si.quad(self.pdf0, h1, h2)[0]  # initial LAI

        fun = lambda X: si.quad(lambda h: self.pdf0(h, lai_mult=X), h1, h2)[0] - LAI
        self.lai_mult = so.fsolve(fun, 0.1)
        if self.lai_mult <= 0:
            print('desired LAI too small')
#        self.lai_mult = 0.04
#        print self.lai_mult

#        print si.quad(self.pdf, h1, h2)[0]  # new LAI


    def f_lw(self, h, lai_mult=1):
        """ """
#        linear = lambda h: (h - self.h1) * 0.07 + self.lad_h1
#        curve = lambda h: np.sqrt((h - self.h1)**2 - 1)
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))

        mult = 1 / (curve(self.hmax) + self.lad_h1) * lai_mult
        return curve(h) * mult + self.lad_h1




    def F_lw(self, h):
        """ cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+sin(1.3+*+(x+-+a)+%2F+(b+-+a))+*+c+%2B+d
        """
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))
        a = self.h1
        b = self.hmax
        c = 1 / (curve(self.hmax) + self.lad_h1) * self.lai_mult
        d = self.lad_h1
        return 1 / 1.3 * c * (a - b) * np.cos(1.3 * (a - h) / (a - b)) + d * h + 0



    def f_up(self, h, lai_mult=1):
        """ """
        mult = self.f_lw(self.hmax, lai_mult) - self.lad_h2
        x = (h - self.hmax) / (self.h2 - self.hmax)
        return ( -(x + 0.5)**2 + (x + 0.5) + 0.75 ) * mult + self.lad_h2


    def find_ub_up(self, lb, lai):
        """
        analytically find upper bound for integration that gives the desired lai
        with lower bound lb

        http://www.wolframalpha.com/input/?i=integrate+from+a+to+b++(-2*(x%2B0.5)%5E2+%2B+2*(x%2B0.5)+%2B+1.5)%2F2+%3D+c+solve+for+b
        """
        return None


    def F_up(self, h):
        """ cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+(-(x+%2B+0.5)**2+%2B+(x+%2B+0.5)+%2B+0.75)+*+a+%2B+b
        """
        x = (h - self.hmax) / (self.h2 - self.hmax)
        dhdx = self.h2 - self.hmax
        a = self.f_lw(self.hmax, self.lai_mult) - self.lad_h2
        b = self.lad_h2
        return x * (-1.0/3 * a * x**2 + a + b) * dhdx


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
        using the lai multipler, integration now gives the desired LAI

        """
        if h >= self.h2:
            return np.nan

        if h >= self.hmax:  # lower portion
            return self.f_up(h, self.lai_mult)

        elif h >= self.h1:  # upper portion
            return self.f_lw(h, self.lai_mult)

        else:
            return np.nan


    def cdf(self, h):
        """ """
        lai_top = self.F_up(self.h2) - self.F_up(self.hmax)
#        print 'lai_top', lai_top

        if h >= self.h2:
            return 0

        elif h >= self.hmax:  # if gets here, is also < self.h_top
            return (self.F_up(self.h2) - self.F_up(h))

        elif h >= self.h1: # if gets here, is also < self.h_max
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

        self.h_tops = np.array([ld['h_top'] for ld in layers])

        h1 = h_bottom
        lad_h1 = 0
        for ld in layers:  # loop through the layer dicts (lds)

            hmax = ld['h_max']
            LAI = ld['fLAI'] * self.LAItot
            h2 = ld['h_top']
            lad_h2 = ld['lad_h_top']

            l = layer(h1, lad_h1, hmax, LAI, h2, lad_h2)  # create instance of layer class

            ld['pdf'] = l.pdf
            ld['cdf'] = l.cdf
            ld['LAI'] = LAI

            h1 = h2
            lad_h1 = lad_h2

        self.lds = layers
        self.LAIlayers = np.array([ld['LAI'] for ld in layers])

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

            return self.lds[lnum]['pdf'](h)


    def cdf(self, h):

        if h > self.h_tops.max():
            return 0

        else:
            lnum = np.where(self.h_tops >= h)[0].min()

            if lnum == len(self.lds) - 1:
                above = 0
            else:
                above = np.sum([self.LAIlayers[lnum+1:]])

            return self.lds[lnum]['cdf'](h) + above


    def inv_cdf(self, ub, lai):
        """
        find the lb that gives that desired lai from numerical integration of the pdf

        using scipy.optimize.fsolve
        """

        fun = lambda lb: si.quad(self.pdf, lb, ub)[0] - lai
        lb = so.fsolve(fun, ub - 1e-6)

        return lb



def distribute_lai_from_cdd(cdd, n):
    """Create LAI profile based on input descriptors of the leaf area distribution.

    Inputs
    ------
    cdd:  canopy description dict. contains many things. see the default CSV for example
    n:    number of layers we want

    """
    LAI = cdd['lai_tot']  # total canopy LAI
    h_bottom = cdd['h_bot'][-1]  # canopy bottom is the bottom of the bottom-most layer
    
    #> form inputs to the layer class
    nlayers = len(cdd['lai_frac'])
    layers = []
    for i in range(nlayers):
        layer = dict(h_max=cdd['h_max_lad'][i],
                     h_top=cdd['h_top'][i], 
                     lad_h_top=cdd['lad_h_top'][i],
                     fLAI=cdd['lai_frac'][i])
        layers.append(layer)
    
    
    cld = canopy_lai_dist(h_bottom, layers[::-1], LAI)  # form dist of leaf area

#    h_canopy = cld.lds[-1]['h_top']  # canopy height is the top of the upper-most layer
    h_canopy = cdd['h_canopy']

    dlai = float(LAI) / (n - 1)  # desired const lai increment
#    print dlai

    lai = np.zeros((n, ))
    z = h_canopy * np.ones_like(lai)  # bottoms of layers?

    ub = h_canopy
    LAIcum = 0
    for i in range(n-2, -1, -1):  # build from top down

        if LAI - LAIcum < dlai:
            assert( i == 0 )
            z[0] = h_bottom
            lai[0] = LAI

        else:
            lb = cld.inv_cdf(ub, dlai)
            z[i] = lb
            lai[i] = lai[i+1] + dlai

            ub = lb
            LAIcum += dlai

    return lai, z



# needs work!
def distribute_lai_gamma(h_c, LAI, n):
    """Create LAI profile using Gamma distribution for leaf area density.

    Inputs
    ------
    h_c : float or int
        canopy height 
    n : int
        number of layers
    LAI : float
        total LAI

    Returns
    -------
    lai, z

    """
    h_max_lad = 0.7*h_c  # could be a param
    d_max_lad = h_c-h_max_lad  # canopy depth

    b = 3.5
    a = d_max_lad/b + 1

    # a = 2.5
    # b = d_max_lad/(a-1)

    g = gamma(a, b)

    #lad = np.zeros((n, ))
    lai = np.zeros((n, ))
    z   = np.zeros((n, ))

    lai_cum_pct = np.linspace(0.99, 0.1, n-2)

    z[-1] = h_c
    z[1:-1] = h_c - g.ppf(lai_cum_pct)  # Gamma ppf depends on depth into the canopy
    z[0] = 0

    lai[-1] = 0
    lai[1:-1] = lai_cum_pct*LAI
    lai[0] = LAI

    return lai, z


def distribute_lai_beta(h_c, LAI, n, h_min=0.5):
    """Create LAI profile using beta distribution for leaf area density.

    """
    h_max_lad = 0.7*h_c
    d_max_lad = h_c - h_max_lad

    # mode is (a-1)/(a+b-2)
    d = d_max_lad/h_c  # relative depth = desired mode
    b = 3
    a = -((b-2)*d + 1)/(d-1)

    #lad = np.zeros((n, ))
    lai = np.zeros((n, ))
    z   = np.zeros((n, ))

    dist = beta(a, b)

    lai_cum_pct = np.linspace(1.0, 0, n)
    # beta pdf is confined to [0, 1] unlike Gamma

    z = (h_c-h_min)*(1-dist.ppf(lai_cum_pct)) + h_min
    # TODO: fix the h_min stuff to be consistent with h_max_lad

    lai = lai_cum_pct*LAI

    return lai, z


# TODO: multi-story beta dist


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    # --------------------------------------------------------------------------------------------------
    # graphical tests of the layer class alone
    plt.close('all')
    
    layer_test = layer(2, 0, 4.5, 1.3, 8, 0.1)
    
    h = np.linspace(1, 9, 200)
    dh = np.insert(np.diff(h), 0, 0)
    dh_step = dh[-1]
    midpts = h[:-1] + dh_step/2
    
    lad0 = np.array([layer_test.pdf0(x) for x in h])
    lad = np.array([layer_test.pdf(x) for x in h])
    lai = np.array([layer_test.cdf(x) for x in h])
    dlai = np.insert(np.diff(lai), 0, 0)
    lai2 = dh_step * np.nancumsum(lad[::-1])[::-1]
    lai3 = np.array([-si.quad(layer_test.pdf, 8, x)[0] for x in midpts])
    
    plt.plot(lad0, h, label='before LAI mult')
    plt.plot(lad, h, label='after')
    plt.plot(-dlai / dh, h, 'm:')
    plt.xlim(xmin=0)
    plt.legend()
    
    plt.figure()
    
    plt.plot(lai, h, c='0.75', lw=4)
    plt.plot(lai2, h, 'm:')
    plt.plot(lai3, midpts, 'b--')  # we see that lai3 gives same as lai
    plt.xlim(xmin=0)
    # --------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # graphical tests of full canopy_lai_dist
    plt.close('all')
    
    l1 = {'h_max': 2.5, 'h_top': 5, 'lad_h_top': 0.1, 'fLAI': 0.2}
    l2 = {'h_max': 8, 'h_top': 13, 'lad_h_top': 0.1, 'fLAI': 0.35}
    l3 = {'h_max': 16, 'h_top': 20, 'lad_h_top': 0.05, 'fLAI': 0.35}
    l4 = {'h_max': 21.5, 'h_top': 23, 'lad_h_top': 0, 'fLAI': 0.1}
    
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
    # -------------------------------------------------------------------------------------------------

    # TODO: graphical test of new Gamma leaf area dist

