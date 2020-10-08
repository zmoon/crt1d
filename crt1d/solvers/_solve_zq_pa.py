"""
Modifications
-------------
(wrt. the verion in pyAPES)

* remove unneeded figure stuff

*

"""
# fmt: off
import numpy as np

# TODO: define kbeam and kdiffuse using the existing machinery
# TODO: redefine input names
# TODO: redefine outputs as dict

#: machine epsilon
EPS = np.finfo(float).eps

short_name = "ZQ-pA"
long_name = "Zhao & Qualls multi-scattering pyAPES"


def kbeam(ZEN, x=1.0):
    """
    COMPUTES BEAM ATTENUATION COEFFICIENT Kb (-) for given solar zenith angle
    ZEN (rad) and leaf angle distribution x (-)
    IN:
        ZEN (rad): solar zenith angle
        x (-): leaf-angle distr. parameter (optional)
                x = 1 : spherical leaf-angle distr. (default)
                x = 0 : vertical leaf-angle distr.
                x = inf. : horizontal leaf-angle distr
    OUT:
        Kb (-): beam attenuation coefficient (array of size(ZEN))
    SOURCE: Campbell & Norman (1998), Introduction to environmental biophysics
    """

    ZEN = np.array(ZEN)
    x = np.array(x)

    XN1 = (np.sqrt(x*x + np.tan(ZEN)**2))
    XD1 = (x + 1.774*(x + 1.182)**(-0.733))
    Kb = XN1 / XD1  # direct beam

    Kb = np.minimum(15, Kb)
    # if Kb<0:
    #     Kb=15

    return Kb

def kdiffuse(LAI, x=1.0):
    """
    COMPUTES DIFFUSE ATTENUATION COEFFICIENT Kd (-) BY INTEGRATING Kd OVER HEMISPHERE
    IN:
        LAI - stand leaf (or plant) are index (m2 m-2)
        x (-): leaf-angle distr. parameter (optional)
                x = 1 : spherical leaf-angle distr. (default)
                x = 0 : vertical leaf-angle distr.
                x = inf. : horizontal leaf-angle distr.
    OUT:
        Kd (-): diffuse attenuation coefficient
    USES:
        kbeam(ZEN, x) for computing beam attenuation coeff.
    SOURCE:
        Campbell & Norman, Introduction to environmental biophysics (1998, eq. 15.5)
    """

    LAI = float(LAI)
    x = np.array(x)

    ang = np.linspace(0, np.pi / 2, 90)  # zenith angles at 1deg intervals
    dang = ang[1]-ang[0]

    # beam attenuation coefficient - call kbeam
    Kb = kbeam(ang, x)
    # print(Kb)

    # integrate over hemisphere to get Kd, Campbell & Norman (1998, eq. 15.5)
    YY = np.exp(-Kb*LAI)*np.sin(ang)*np.cos(ang)

    Taud = 2.0*np.trapz(YY*dang)
    Kd = -np.log(Taud) / (LAI + EPS)  # extinction coefficient for diffuse radiation

    return Kd


def solve_zq_pa(
    lai,
    Clump,
    x,
    ZEN,
    IbSky,
    IdSky,
    LeafAlbedo,
    SoilAlbedo,
):
    """
    Computes incident (Wm-2 ground) SW radiation and absorbed (Wm-2 (leaf) radiation within canopies.
    INPUT:
        LAIz: layewise one-sided leaf-area index (m2m-2)
        Clump: element clumping index (0...1)
        x: param. of leaf angle distribution
            (1=spherical, 0=vertical, inf.=horizontal) (-)
        ZEN: solar zenith angle (rad),scalar
        IbSky: incident beam radiation above canopy (Wm-2),scalar
        IdSky: downwelling diffuse radiation above canopy (Wm-2),scalar
        LAI: leaf (or plant-area) index, 1-sided (m2m-2),scalar
        LeafAlbedo: leaf albedo of desired waveband (-)
        SoilAlbedo: soil albedo of desired waveband (-)
        PlotFigs="True" plots figures.
    OUTPUT:
        SWbo: direct SW at z (Wm-2(ground))
        SWdo: downwelling diffuse at z (Wm-2(ground))
        SWuo: upwelling diffuse at z (Wm-2(ground))
        Q_sl: incident SW normal to sunlit leaves (Wm-2)
        Q_sh: incident SW normal to shaded leaves (Wm-2)
        q_sl: absorbed SW by sunlit leaves (Wm-2(leaf))
        q_sh: absorbed SW by shaded leaves (Wm-2(leaf))
        q_soil: absorbed SW by soil surface (Wm-2(ground))
        f_sl: sunlit fraction of leaves (-): Note: to get sunlit fraction below
            all vegetation f_sl[0] / Clump
        alb: canopy albedo (-)
    USES:
        kbeam(ZEN,x) & kdiffuse(LAI,x=1) for computing beam and diffuse attenuation coeff
    SOURCE:
        Zhao W. & Qualls R.J. (2005). A multiple-layer canopy scattering model
        to simulate shortwave radiation distribution within a homogenous plant
         canopy. Water Resources Res. 41, W08409, 1-16.
    NOTE:
        At least for conifers NIR LeafAlbedo has to be decreased from leaf-scale  values to
        correctly model canopy albedo of clumped canopies.
        Adjustment from ~0.7 to 0.55 seems to be sufficient. This corresponds roughlty to
        a=a_needle*[4*STAR / (1- a_needle*(1-4*STAR))], where a_needle is needle albedo
        and STAR silhouette to total area ratio of a conifer shoot. STAR ~0.09-0.21 (mean 0.14)
        for Scots pine (Smolander, Stenberg et al. papers)
    CODE:
    Samuli Launiainen, Luke. Converted from Matlab & tested 15.5.2017
    """
    # --- check inputs and create local variables
    IbSky = max(IbSky, 0.0001)
    IdSky = max(IdSky, 0.0001)

    # original and computational grid
    # // LAI = Clump*sum(LAIz)  # effective LAI, corrected for clumping (m2 m-2)
    # // Lo = Clump*LAIz  # effective layerwise LAI (or PAI) in original grid

    # instead, assume that effective *cumulative* LAI has been passed in;
    # use Clump only for the absorption stuff
    LAI = lai[0]  # effective LAI, corrected for clumping (m2 m-2)
    # `lai` input

    # cumulative plant area index: node 0 is canopy bottom, N is top
    # this is the same as the `lai` input
    Lcumo = lai

    # --- create computation grid
    N = np.size(lai)  # nr of original layers
    M = np.minimum(100, N)  # nr of comp. layers
    L = np.ones([M+2])*LAI / M  # effective leaf-area density (m2m-3)
    L[0] = 0.
    L[M + 1] = 0.
    Lcum = np.cumsum(np.flipud(L), 0)  # cumulative plant area from top

    # ---- optical parameters
    aL = np.ones([M+2])*(1 - LeafAlbedo)  # leaf absorptivity
    tL = np.ones([M+2])*0.4  # transmission as fraction of scattered radiation
    rL = np.ones([M+2])*0.6  # reflection as fraction of scattered radiation

    # soil surface, no transmission
    aL[0] = 1. - SoilAlbedo
    tL[0] = 0.
    rL[0] = 1.

    # upper boundary = atm. is transparent for SW
    aL[M+1] = 0.
    tL[M+1] = 1.
    rL[M+1] = 0.

    # black leaf extinction coefficients for direct beam and diffuse radiation
    Kb = kbeam(ZEN, x)
    Kd = kdiffuse(LAI, x)

    # fraction of sunlit & shad ground area in a layer (-)
    f_sl = np.flipud(np.exp(-Kb*(Lcum)))

    # beam radiation at each layer
    Ib = f_sl*IbSky

    # propability for beam and diffuse penetration through layer without interception
    taub = np.zeros([M+2])
    taud = np.zeros([M+2])
    taub[0:M+2] = np.exp(-Kb*L)
    taud[0:M+2] = np.exp(-Kd*L)

    # soil surface is non-transparent
    taub[0] = 0.
    taud[0] = 0.

    # backward-scattering functions (eq. 22-23) for beam rb and diffuse rd
    rb = np.zeros([M+2])
    rd = np.zeros([M+2])
    rb = 0.5 + 0.3334*(rL - tL) / (rL + tL)*np.cos(ZEN)
    rd = 2.0 / 3.0*rL/(rL + tL) + 1.0 / 3.0*tL / (rL + tL)

    rb[0] = 1.
    rd[0] = 1.
    rb[M+1] = 0.
    rd[M+1] = 0.

    # --- set up tridiagonal matrix A and solve SW without multiple scattering
    # from A*SW = C (Zhao & Qualls, 2006. eq. 39 & 42)
    A = np.zeros([2*M+2, 2*M+2])

    # lowermost rows: 0 = soil surface, M+2 is upper boundary
    A[0, 0] = 1.

    A[1, 0] = - (taud[1] + (1 - taud[1])*(1 - aL[1])*(1 - rd[1]))
    A[1, 1] = - 1*(taud[1] + (1 - taud[1])*(1 - aL[1])*(1 - rd[1]))*(1 - aL[0])
    A[1, 2] = (1 - 1*rd[1]*(1 - aL[0])*1*(1 - aL[1])*(1 - taud[1]))

    # middle rows
    for k in range(1, M + 1):
        A[2*k-1, 2*k-2] = - (taud[k] + (1 - taud[k])*(1 - aL[k])*(1 - rd[k]))
        A[2*k-1, 2*k-1] = - rd[k - 1]*(taud[k] + (1 - taud[k])*(1 - aL[k])*(1 - rd[k]))*(1 - aL[k-1])*(1 - taud[k-1])
        A[2*k-1, 2*k] = (1 - rd[k-1]*rd[k]*(1 - aL[k-1])*(1 - taud[k-1])*(1 - aL[k])*(1 - taud[k]))

        A[2*k, 2*k-1] = (1 - rd[k]*rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        A[2*k, 2*k] = - rd[k+1]*(taud[k] + (1 - taud[k])*(1 - aL[k])*(1 - rd[k]))*(1 - aL[k+1])*(1 - taud[k+1])
        A[2*k, 2*k+1] = -(taud[k] + (1 - taud[k])*(1 - aL[k])*(1 - rd[k]))

    # uppermost node2*M+2
    A[2*M+1, 2*M+1] = 1.
    del k
    # print A

    # --- RHS vector C
    C = np.zeros([2*M+2, 1])

    # lowermost row
    C[0] = SoilAlbedo*Ib[0]
    n = 1  # dummy
    for k in range(1, M+1):  # k=2:M-1,
        C[n] = (1 - rd[k-1]*rd[k]*(1 - aL[k-1])*(1 - taud[k-1])*(1 - aL[k])*(1 - taud[k]) )*rb[k]*(1 - taub[k])*(1 - aL[k])*Ib[k]
        C[n+1] = (1 - rd[k]*rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))*(1 - taub[k])*(1 - aL[k])*(1 - rb[k])*Ib[k]
        # Ib(k+1) Ib(k):n sijaan koska tarvitaan kerrokseen tuleva
        n = n + 2

    # uppermost row
    C[2*M+1] = IdSky

    # ---- solve A*SW = C
    SW = np.linalg.solve(A, C)

    # upward and downward hemispherical radiation (Wm-2 ground)
    SWu0 = SW[0:2*M+2:2]
    SWd0 = SW[1:2*M+2:2]
    del A, C, SW, k

    # ---- Compute multiple scattering, Zhao & Qualls, 2005. eq. 24 & 25.
    # downwelling diffuse after multiple scattering, eq. 24
    SWd = np.zeros([M+1])
    for k in range(M-1, -1, -1):  # downwards from layer k+1 to layer k
        X = SWd0[k+1] / (1 - rd[k]*rd[k+1]*(1-aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        Y = SWu0[k]*rd[k+1]*(1 - aL[k+1])*(1 - taud[k+1]) / (1 - rd[k]*rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        SWd[k+1] = X + Y
    SWd[0] = SWd[1] # SWd0[0]
    # print SWd

    # upwelling diffuse after multiple scattering, eq. 25
    SWu = np.zeros([M+1])
    for k in range(0, M, 1):  # upwards from layer k to layer k+1
        X = SWu0[k] / (1 - rd[k]*rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        Y = SWd0[k+1]*rd[k]*(1 - aL[k])*(1 - taud[k]) / (1 - rd[k]*rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        SWu[k] = X + Y
    SWu[M] = SWu[M-1]

    # match dimensions of all vectors
    Ib = Ib[1:M+2]
    f_sl = f_sl[1:M+2]
    Lcum = np.flipud(Lcum[0:M+1])
    aL = aL[0:M+1]

    # --- NOW return values back to the original grid
    f_slo = np.exp(-Kb*(Lcumo))
    SWbo = f_slo*IbSky  # Beam radiation


    # interpolate diffuse fluxes
    X = np.flipud(Lcumo)
    xi = np.flipud(Lcum)
    SWdo = np.flipud(np.interp(X, xi, np.flipud(SWd)))
    SWuo = np.flipud(np.interp(X, xi, np.flipud(SWu)))
    del X, xi

    # incident radiation on sunlit and shaded leaves Wm-2
    Q_sh = Clump*Kd*(SWdo + SWuo)  # normal to shaded leaves is all diffuse
    Q_sl = Kb*IbSky + Q_sh  # normal to sunlit leaves is direct and diffuse

    # absorbed components
    aLo = np.ones(len(Lo))*(1 - LeafAlbedo)
    aDiffo = aLo*Kd*(SWdo + SWuo)
    aDiro = aLo*Kb*IbSky

    # stand albedo
    alb = SWuo[-1] / (IbSky + IdSky + EPS)
    # print alb
    # soil absorption (Wm-2 (ground))
    q_soil = (1 - SoilAlbedo)*(SWdo[0] + SWbo[0])

    # correction to match absorption-based and flux-based albedo, relative error <3% may occur
    # in daytime conditions, at nighttime relative error can be larger but fluxes are near zero
    aa = (sum(aDiffo*Lo + aDiro*f_slo*Lo) + q_soil) / (IbSky + IdSky + EPS)
    F = (1. - alb) / aa
    # print('F', F)
    if F <= 0 or np.isfinite(F) is False:
        F = 1.

    aDiro = F*aDiro
    aDiffo = F*aDiffo
    q_soil = F*q_soil

    # sunlit fraction in clumped foliage; clumping means elements shade each other
    f_slo = Clump*f_slo

    # Following adjustment is required for energy conservation, i.e. total absorbed radiation
    # in a layer must equal difference between total radiation(SWup, SWdn) entering and leaving the layer.
    # Note that this requirement is not always fullfilled in canopy rad. models.
    # now sum(q_sl*f_slo* + q_sh*(1-f_slo)*Lo = (1-alb)*(IbSky + IdSky)
    q_sh = aDiffo*Clump  # shaded leaves only diffuse
    q_sl = q_sh + aDiro  # sunlit leaves diffuse + direct

    return SWbo, SWdo, SWuo, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_slo, alb
