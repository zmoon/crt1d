"""
Create spectral input data using external packages.
"""


def leaf_ps5():
    """Run PROSPECT from Python package ``prosail`` to generate leaf spectra."""
    import prosail

    raise NotImplementedError


def solar_sp2(dt, lat, lon, *, temp=25, pres=1000, utcoffset=0):
    """Run SPCTRAL2 from Python package ``solar_utils`` (``SolarUtils`` on PyPI).

    Parameters
    ----------
    dt : datetime.datetime
        In UTC, but with timezone not specified. (UTC offset specified separately!)
    lat, lon : float
        Latitude and longitude of the location (deg.).

    utcoffset : int
        UTC offset, e.g., -5 for US Eastern Standard Time (EST)

    temp : float
        Air temperature at the location (deg. C).
    pres : float
        Amospheric surface pressure at the location (hPa/mb).

    """
    import solar_utils

    # construct inputs for solar_utils.spectrl2
    units = 1
    location = [lat, lon, utcoffset]
    list_datetime = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
    weather = [pres, temp]

    # run SOLPOS to get angles
    (zenith, azimuth), _ = solar_utils.solposAM(location, list_datetime, weather)

    # run SPCTRAL2 (following their readme example)
    specdif, specdir, specetr, specglo, specx = solar_utils.spectrl2(
        units, location, list_datetime, weather, orientation, atmospheric_conditions, albedo
    )

    # construct dataset
    # with time variable and solar angles!

    raise NotImplementedError
