"""
Create spectral input data using external packages.
"""
import warnings

import numpy as np
import xarray as xr

from ..variables import _tup
from ..variables import _wl_coord_dict


def leaf_ps5(n=1.2, cab=30.0, car=10.0, cbr=1.0, ewt=0.015, lma=0.009):
    """Run PROSPECT-5 from Python package ``prosail``
    (source `on GitHub <https://github.com/jgomezdans/prosail>`_)
    to generate leaf spectra.

    Default values are the same as those on the
    `online PROSPECT page <http://opticleaf.ipgp.fr/index.php?page=prospect>`_.

    Parameters
    ----------
    n : float
        Leaf structure parameter (number of effective layers of leaf?; dimensionless).
        Typical range: [0.8, 3.0].
    cab : float
        Chlorophyll a+b concentration (μg cm-2).
        Typical range: [0, 100].
    car : float
        Carotenoid concentration (μg cm-2).
        Typical range: [0, 25].
    cbr : float
        Brown pigment fraction/factor in [0, 1].
    ewt : float
        Equivalent leaf water thickness (cm).
        Typical range: [0, 0.05].
    lma : float
        Leaf dry mass per unit area (g cm-2).
        Typical range: [0, 0.02].

    Returns
    -------
    xr.Dataset
        Dataset of the leaf reflectance and transmittance spectra,
        and PROSPECT input parameters as dimensionless variables.

    Notes
    -----
    Quoted typical ranges are based on the PROSPECT page and Python PROSAIL readme linked above.
    """
    import prosail

    wl_nm, r, t = prosail.run_prospect(n, cab, car, cbr, ewt, lma, prospect_version="5")

    wl = wl_nm / 1000  # nm -> um

    attrs = {}
    ds = xr.Dataset(
        coords=_wl_coord_dict(wl),
        data_vars={
            "rl": _tup("leaf_r", r),
            "tl": _tup("leaf_t", t),
            "n": ((), n, {"long_name": "Leaf structure parameter", "units": ""}),
            "cab": ((), cab, {"long_name": "Chlorophyll a+b", "units": "μg cm-2"}),
            "car": ((), car, {"long_name": "Carotenoid", "units": "μg cm-2"}),
            "cbr": ((), cbr, {"long_name": "Brown pigment", "units": ""}),
            "ewt": ((), ewt, {"long_name": "Equivalent leaf water thickness", "units": "cm"}),
            "lma": ((), lma, {"long_name": "Leaf dry mass density", "units": "g cm-2"}),
        },
        attrs=attrs,
    )
    return ds


def solar_sp2(
    dt,
    lat,
    lon,
    *,
    utcoffset=0,
    #
    temp=25,
    pres=1000,
    #
    ozone=None,
    tau500=0.2,
    watvap=1.36,
    #
    alpha=1.14,
    assym=0.65,
    #
    albedo=None,
):
    """Run SPCTRAL2 using `SolarUtils <https://github.com/SunPower/SolarUtils>`_
    (Python package name ``solar_utils``).

    The defaults are mostly consistent with :func:`solar_utils.spectrl2`,
    but some of the inputs are in a different style.

    Parameters
    ----------
    dt : datetime.datetime
        In UTC, but with timezone not specified. (UTC offset specified separately!)
    lat, lon : float
        Latitude and longitude of the location (deg.). Longitude in -180--180.
    utcoffset : int
        UTC offset, e.g. -5 for US Eastern Standard Time (EST)
    temp : float
        Air temperature at the location (deg. C).
    pres : float
        Amospheric surface pressure at the location (hPa/mb).
    ozone : float
        Total column ozone (cm). The default is to use SPCTRAL2's internal parameterization
        based on location and time of day/year (Heuklon 1978). Passing ``-1.0`` will have
        the same result.
    tau500 : float
        Aerosol optical depth (base e) at 500 nm.
        Typical range: [0.05, 0.55] for clear sky.
        Should be in [0, 10].
    watvap : float
        Total column precipitable water vapor (cm).
        Typical range: [0.3, 6].
        Should be in [0, 100].
    alpha : float
        Power on Angstrom turbidity. If you pass a negative number, you will get ``1.14``.
    assym : float
        Aerosol asymmetry factor. In the C code they say this is a typical rural value.
        Should be in (0, 1). If you pass ``-1.0``, you will get 0.65.
    albedo : dict
        keys: wavelength (μm), values: albedo at that wavelength; length 6.
        The default is 0.2 (across the board) for wavelengths
        ``[0.3, 0.7, 0.8, 1.3, 2.5, 4.0]``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the spectra, solar zenith angle, and time/location info.
    """
    import solar_utils

    # tilt and aspect (azimuth angle) of collector panel
    # aspect: S=180, N=0, E=90, W=270
    # tilt: pass `lat` for tilted at latitude; note Sun-tracking assumed if pass negative number
    # note that tilt and aspect don't affect the direct normal calculation
    tilt = 0
    aspect = 0  # doesn't matter for tilt=0; `180 if lat >= 0 else 0` to tilt towards Sun

    # by default, send flag to activate the internal ozone parameterization
    if ozone is None:
        ozone = -1.0

    # construct inputs for solar_utils.spectrl2
    # variable names mostly following the `solar_utils.spectrl2` argument names
    units = 1  # W/m2/micron
    location = [lat, lon, utcoffset]
    list_datetime = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
    weather = [pres, temp]
    orientation = [tilt, aspect]
    atmospheric_conditions = [alpha, assym, ozone, tau500, watvap]

    if albedo is None:
        albedo = dict(zip([0.3, 0.7, 0.8, 1.3, 2.5, 4.0], [0.2] * 6))
        # note that passing albedo=[-1] to spectrl2 will activate its internal default
    assert len(albedo) == 6, "`albedo` needs to be passed as a length-6 dict"

    # run SOLPOS to get angles (in deg.)
    # (SPCTRAL2 also runs it but whatevs)
    (zenith, azimuth), _ = solar_utils.solposAM(location, list_datetime, weather)
    mu = np.cos(np.deg2rad(zenith))
    if zenith > 90:
        warnings.warn("computed solar zenith angle > 90 deg.")

    # run SPCTRAL2 (following SolarUtils readme example)
    specdif, specdir, specetr, specglo, specx = solar_utils.spectrl2(
        units, location, list_datetime, weather, orientation, atmospheric_conditions, albedo
    )

    specdir = np.array(specdir)  # direct, *not* downward hemispherical, but direct normal
    specdif = np.array(specdif)  # diffuse, downward hemispherical *wrt. panel surface*
    specglo = np.array(specglo)  # global on panel
    # from the C code, specglo is calculated as specdir*cosinc + specdif
    # https://github.com/SunPower/SolarUtils/blob/6ea5aca2e354dfec980201e84c7812a305c5eb25/solar_utils/src/spectrl2_2.c#L448
    # where, cosinc is the cosine of the inclination angle wrt. the collector
    # this seems like the wrong equation (or abbreviated),
    # since the equation 3-18 they reference is much longer
    cosinc_all = (specglo - specdif) / specdir
    cosinc = np.nanmean(cosinc_all)
    # assert np.allclose(cosinc, cosinc_all)
    inc = np.rad2deg(np.arccos(cosinc), dtype=np.float64)
    # should be equal to sza for tilt=0, but one might be refraction-corrected
    assert np.isclose(inc, zenith, rtol=1e-3), "SZA and incidence angle should be = for tilt=0"

    # for tilt=0, we should have specglo and specdif+specdir equal
    assert np.isclose(
        np.nansum(specglo), mu * np.nansum(specdir) + np.nansum(specdif), rtol=1e-3
    ), "specglo should be = mu*specdir + specdif for tilt=0"

    specunits = "W m-2 μm-1"  # spectral units

    return xr.Dataset(
        coords={"wl": _tup("wl", specx)},
        data_vars={
            "SI_dr": (
                "wl",
                specdir * mu,
                {"units": specunits, "long_name": "Downward direct spectral irradiance"},
            ),
            "SI_df": (
                "wl",
                specdif,
                {"units": specunits, "long_name": "Downward diffuse spectral irradiance"},
            ),
            "SI_et": (
                "wl",
                specetr,
                {"units": specunits, "long_name": "Extraterrestrial spectral irradiance"},
            ),
            #
            "sza": _tup("sza", zenith),
            "saa": ((), azimuth, {"long_name": "Solar azimuth angle", "units": "deg"}),
            "inc": (
                (),
                inc,
                {"long_name": "Direct beam incidence angle on collector panel", "units": "deg"},
            ),
            "time": ((), dt),
            "lat": ((), float(lat), {"long_name": "Latitude", "units": "deg"}),
            "lon": ((), float(lon), {"long_name": "Longitude", "units": "deg"}),
        },
    )
