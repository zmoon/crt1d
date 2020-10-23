"""
Create spectral input data using external packages.
"""
import numpy as np
import xarray as xr

from ..spectra import interpret_spectrum
from ..variables import _tup


def leaf_ps5():
    """Run PROSPECT from Python package ``prosail`` to generate leaf spectra."""
    import prosail

    raise NotImplementedError


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
    """Run SPCTRAL2 from Python package ``solar_utils`` (``SolarUtils`` on PyPI).

    The defaults are mostly consistent with `solar_utils.spectrl2`,
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
        Should be in [0, 10]
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
    """
    import solar_utils

    # tilt and aspect (azimuth angle) of collector panel
    # aspect: S=180, N=0, E=90, W=270
    # for tilt=lat, doesn't affect direct, only diffuse and global horizontal
    tilt = lat  # tilted at latitude; note Sun-tracking assumed if pass negative number
    aspect = 180 if lat >= 0 else 0
    # 180 is listed as the default in the original C code, which has an example for NH
    # let's hope this gives "collector" as local surface plane

    if ozone is None:
        ozone = -1.0  # flag to activate the internal ozone parameterization

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

    # run SOLPOS to get angles
    # (SPCTRAL2 also runs it but whatevs)
    (zenith, azimuth), _ = solar_utils.solposAM(location, list_datetime, weather)

    # run SPCTRAL2 (following their readme example)
    specdif, specdir, specetr, specglo, specx = solar_utils.spectrl2(
        units, location, list_datetime, weather, orientation, atmospheric_conditions, albedo
    )

    specdir = np.array(specdir)
    specdif = np.array(specdif)
    specglo = np.array(specglo)  # not necessarily equiv. to dir+dif

    specunits = "W m-2 μm-1"  # spectral units

    return xr.Dataset(
        coords={"wl": _tup("wl", specx)},
        data_vars={
            "SI_dr": (
                "wl",
                specdir,
                {"units": specunits, "long_name": "Spectral direct irradiance"},
            ),
            "SI_df": (
                "wl",
                specdif,
                {"units": specunits, "long_name": "Spectral diffuse irradiance"},
            ),
            "SI_gl": (
                "wl",
                specglo,
                {"units": specunits, "long_name": "Global horizontal spectral irradiance"},
            ),
            "SI_et": (
                "wl",
                specetr,
                {"units": specunits, "long_name": "Extraterrestrial spectral irradiance"},
            ),
            #
            "sza": _tup("sza", zenith),
            "saa": ((), azimuth, {"long_name": "Solar azimuth angle", "units": "deg"}),
            "time": ((), dt),
            "lat": ((), float(lat), {"long_name": "Latitude", "units": "deg"}),
            "lon": ((), float(lon), {"long_name": "Longitude", "units": "deg"}),
        },
    )
