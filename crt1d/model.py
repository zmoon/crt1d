"""
This module contains the model class, which can be used to conveniently solve CRT problems
using different solvers with minimal boilerplate code needed.
(At least that is the goal.)
"""
# from dataclasses import dataclass
import warnings
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .cases import load_default_case
from .diagnostics import (
    calc_leaf_absorption,
)
from .solvers import AVAILABLE_SCHEMES
from .solvers import RET_KEYS_ALL_SCHEMES  # the ones all schemes must return


CANOPY_DESCRIPTION_KEYS = [
    "lai",
    "z",
    "dlai",
    "lai_tot",
    "lai_eff",
    "mla",
    "clump",
    "leaf_t",
    "leaf_r",
    "soil_r",
    "wl_leafsoil",
    "orient",  # don't really need both this and mla as input
    "G_fn",
]

# class for displaying canopy parameters/data (model paramters/inputs, not outputs)
CanopyDescription = namedtuple("CanopyDescription", " ".join(k for k in CANOPY_DESCRIPTION_KEYS))


class Model:
    """A general class for testing 1-D canopy radiative transfer schemes.

    Optional keyword arguments are used to create the cnpy_descrip and initial cnpy_rad_state
    dicts that will be used to pass info to the solvers. If not provided, defaults are used.
    """

    #
    required_input_keys = (
        "lai",
        "z",
        "mla",
        "clump",
        "leaf_t",
        "leaf_r",
        "soil_r",
        "wl_leafsoil",
        "orient",  # don't really need both this and mla as input
        "G_fn",
        "I_dr0_all",
        "I_df0_all",  # spectral (W/m^2/um)
        "wl",
        "dwl",  # for the toc spectra
        "psi",
    )
    """
    Required inputs
    """

    _schemes = AVAILABLE_SCHEMES

    def __init__(
        self,
        scheme="2s",
        nlayers=60,
        # *,
        # savd_id=None,
    ):
        """
        Create a model instance based on the default setup.

        Parameters
        ----------
        scheme : str
            Identifier for the desired canopy radiative transfer scheme.
        psi : float
            Solar zenith angle (radians).
        nlayers : int
            Number of in-canopy layers to use in the solver (interface levels).
        """
        # load default case, for given nlayers
        self.nlayers = nlayers
        self.p_default = load_default_case(nlayers=self.nlayers)
        # initial settings based on default
        self._p = deepcopy(self.p_default)

        # assign scheme
        self.assign_scheme(scheme)  # assigns scheme info dict to self.scheme

        # check inputs
        self._check_inputs()

        # run/output variables
        self._run_count = 0  # TODO: store last_state?
        self.absorption = None  # initially no absorption data
        self.out = {}  # standard scheme outputs
        self.out_extra = {}  # extra outputs

    @property
    def p(self):
        """Model parameters."""
        print(f"Please update parameters using `.update_p()`!")

    @property
    def cd(self):
        """Canopy description."""
        p = self._p
        return CanopyDescription(**{k: v for k, v in p.items() if k in CANOPY_DESCRIPTION_KEYS})

    def __repr__(self):
        scheme_name = self.scheme["name"]
        psi = self._p["psi"]
        return f"Model(scheme={scheme_name!r}, psi={psi:.4g})"

    def assign_scheme(self, scheme_name, *, verbose=False):
        """Using the :const:`.solvers.AVAILABLE_SCHEMES` dict,
        assign scheme and necessary scheme attrs
        """
        schemes = AVAILABLE_SCHEMES
        scheme_names = list(schemes.keys())
        try:
            self.scheme = schemes[scheme_name]
            if verbose:
                print("\n\n")
                print("=" * 40)
                print("scheme:", self.scheme["name"])
                print("-" * 40)
        except KeyError:
            print(f"{scheme_name!r} is not a valid scheme name/ID!")
            print(f"The valid ones are: {', '.join(scheme_names)}.")
            print("Defaulting to Dickinson-Sellers two-stream.\n")
            self.scheme = schemes["2s"]
            # also could self.terminate() or yield error or set flag

        return self  # for chaining

    def update_p(self, **kwargs):
        """
        Parameters
        ----------
        `**kwargs`
            Used to update the model parameters.
        """
        import traceback

        p0 = deepcopy(self._p)
        try:
            for k, v in kwargs.items():
                if k not in Model.required_input_keys:
                    warnings.warn(f"{k!r} is not intended as an input and will be ignored")
                    continue
                # else, it is intended to be an input, so try to use it
                self._p[k] = v

            # now update other parameters and validate
            self._check_inputs()  # checks self._p

        except Exception:  # AssertionError or other failure in calculating new derived params
            warnings.warn(
                f"Updating parameters failed. "
                f"Full traceback:\n\n{traceback.format_exc()}\n"
                "Reverting."
            )
            self._p = p0  # undo

        return self  # for chaining

    def _check_inputs(self):
        """
        Check input LAI profile and compute additional vars from it...,
        Check wls match optical props and toc spectra...

        update some class var (e.g. nlayers)
        """
        p = self._p
        # check for required input all at once variables
        for key in Model.required_input_keys:
            if key not in p:
                raise Exception(f"required key {key} is not present. Set it using `update_p`.")

        # lai profile
        lai = p["lai"]
        z = p["z"]
        dz = np.diff(z)
        zm = z[:-1] + 0.5 * dz  # midpts
        assert (
            z.size == lai.size
        )  # z values go with lai (the cumulative LAI profile; interface levels)
        self.nlev = lai.size
        assert z[-1] > z[0]  # z increasing
        assert lai[0] > lai[-1]  # LAI decreasing
        assert lai[-1] == 0
        p["lai_tot"] = lai[0]
        p["lai_eff"] = lai * p["clump"]
        dlai = lai[:-1] - lai[1:]
        assert dlai.sum() == lai[0]
        p["dlai"] = dlai
        p["dlai_eff"] = dlai * p["clump"]
        p["zm"] = zm  # z for dlai, layer centers
        p["dz"] = dz
        assert dlai.size == zm.size and zm.size == dz.size

        # solar zenith angle, giving precedence to psi if have both mu and psi already
        psi = p["psi"]
        try:
            mu = p["mu"]
            if mu != np.cos(psi):
                warnings.warn(
                    "Provided `mu` not consistent with provided `psi`. "
                    "`mu` will be updated based on the value of `psi`."
                )
        except KeyError:  # no mu
            p["mu"] = np.cos(psi)

        # TODO: check mla and orient/x, similar to psi/mu

        # check the two wl sources
        wl_toc = p["wl"]  # for the toc spectra
        wl_op = p["wl_leafsoil"]  # for optical properties
        assert wl_toc.size == wl_op.size
        if not np.allclose(wl_toc, wl_op):
            # print('wl for optical props and toc BC appear to be incompatible')
            warnings.warn(
                "Provided wavelengths for optical props (`wl_leafsoil`) and toc BC (`wl`) "
                "appear to be incompatible:\n"
                f"`wl - wl_leafsoil`:\n{wl_toc-wl_op}"
            )
            # or could convert, but which to choose as base?
        self.nwl = wl_toc.size  # precedence to the toc spectra one as definition

        # G_fn and K_b_fn ?
        # p['G_fn'] = p['G_fn']
        p["K_b_fn"] = lambda psi_: p["G_fn"](psi_) / np.cos(psi_)
        p["G"] = p["G_fn"](psi)
        p["K_b"] = p["K_b_fn"](psi)
        # ^ should clumping index be included somewhere here?

    def run(self, **extra_solver_kwargs):
        """
        TODO: could add verbose option
        """
        self._run_count += 1

        # check wavelengths are compatible etc.
        # should already have been done at least once by now, but whatever
        self._check_inputs()

        # construct dict of kwargs to pass to the solver
        d = {**self._p, **extra_solver_kwargs}
        scheme = self.scheme
        args = {k: d[k] for k in scheme["args"]}

        # run
        sol = scheme["solver"](**args)

        # use the dict returned by the solver to update our state
        self.out.update({k: v for k, v in sol.items() if k in RET_KEYS_ALL_SCHEMES})
        self.out_extra.update(
            {f"{k}_scheme": v for k, v in sol.items() if k not in RET_KEYS_ALL_SCHEMES}
        )

        return self  # for chaining

    @property
    def out_all(self):
        """Standard and extra outputs."""
        return {**self.out, **self.out_extra}

    def calc_absorption(self, *, bands="all"):
        """Calculate layerwise absorption variables using routines in module diagnostics."""
        if self._run_count == 0:
            raise Exception("Must run the model first.")

        absorption = calc_leaf_absorption(
            self._p,
            self.out,
            band_names_to_calc=bands,
        )
        # update model attr
        self.absorption = absorption

        return self  # for chaining

    def to_xr(self, *, info=""):
        """Construct an `xarray.Dataset`."""
        if self._run_count == 0:
            raise Exception("Must run the model before creating the dataset.")
        p = self._p
        out = self.out_all
        # canopy descrip
        lai = p["lai"]
        dlai = p["dlai"]
        z = p["z"]  # at lai values (layer interfaces)
        zm = p["zm"]
        # wavelength grid
        wl = p["wl"]
        dwl = p["dwl"]
        # radiation geometry/setup
        psi = p["psi"]
        mu = p["mu"]
        sza = np.rad2deg(psi)
        G = p["G"]
        K_b = p["K_b"]
        # scheme info
        scheme_lname = self.scheme["long_name"]
        scheme_sname = self.scheme["short_name"]
        scheme_name = self.scheme["name"]
        # canopy RT solution
        Idr = out["I_dr"]
        Idfd = out["I_df_d"]
        Idfu = out["I_df_u"]
        F = out["F"]
        crds = ["z", "wl"]
        crds2 = ["zm", "wl"]

        # > create dataset
        ln = "long_name"
        z_units = dict(units="m")
        wl_units = dict(units="μm")
        E_units = dict(units="W m-2")  # E (energy flux) units
        # SE_units = dict(units="W m-2 um-1")  # spectral E units
        pf_units = dict(units="μmol photons m-2 s-1")  # photon flux units
        lai_units = dict(units="(m2 leaf) m-2")

        # construct data vars for absorption (many)
        # allow abs not calculated or abs from scheme
        abs_scheme = {}  # {f"{k}_scheme": v for k, v in self.p.items() if k[:2] == "aI"}
        abs_post = self.absorption if self.absorption is not None else {}
        absorption = {**abs_scheme, **abs_post}  # merge
        abs_data_vars = {}  # collect data_vars tuples here
        for k, v in absorption.items():
            # energy or photon units: irradiance or PFD
            if "I_" in k or k == "aI":
                baseq = "irradiance"
                units = E_units
            elif "PFD" in k:
                baseq = "PFD"
                units = pf_units
            else:
                warnings.warn(f"key {k!r} not identified as either PFD or irradiance")
                baseq = ""
            # sunlit, shaded, or both
            if "_sl" in k:
                by = " by sunlit leaves"
            elif "_sh" in k:
                by = " by shaded leaves"
            else:
                by = ""
            # specific band or spectral
            if any(w in k for w in ["_PAR", "_solar", "_NIR", "_UV"]):  # need to fix
                bn = k.split("_")[-1]
                band = f"{bn} "
                # if 'aI' in k:
                if k[0] == "a":
                    crds_ = ["zm"]
                else:
                    crds_ = ["z"]
            else:
                band = ""
                if k[0] == "a":  # absorbed is on mid levels
                    crds_ = crds2
                else:
                    crds_ = crds  # radiative fluxes are on interface levels
            # direct or diffuse (up or down)
            if "_dr" in k:
                dfdr = "direct "
            elif "_df_u" in k:
                dfdr = "upward diffuse "
            elif "_df_d" in k:
                dfdr = "downward diffuse "
            else:
                dfdr = ""
            # absorbed or not
            if k[0] == "a":
                absorbed = "absorbed "
            else:
                absorbed = ""
            # construct long_name
            lni = f"{absorbed}{dfdr}{band}{baseq}{by}"
            # construct data_vars tuple
            abs_data_vars[k] = (
                crds_,
                v,
                {
                    **units,
                    ln: lni,
                },
            )

        # print(abs_data_vars)

        dset = xr.Dataset(
            coords={
                "z": ("z", z, {**z_units, ln: "height above ground"}),
                "wl": ("wl", wl, {**wl_units, ln: "wavelength"}),
                "zm": ("zm", zm, {**z_units, ln: "layer midpoint height"}),
            },
            data_vars={
                "I_dr": (crds, Idr, {**E_units, ln: "direct beam irradiance (binned)"}),
                "I_df_d": (crds, Idfd, {**E_units, ln: "downward diffuse irradiance (binned)"}),
                "I_df_u": (crds, Idfu, {**E_units, ln: "upward diffuse irradiance (binned)"}),
                "F": (crds, F, {**E_units, ln: "actinic flux (binned)"}),
                "I_d": (crds, Idr + Idfd, {**E_units, ln: "downward irradiance (binned)"}),
                "dwl": ("wl", dwl, {**wl_units, ln: "wavelength band width"}),
                "lai": ("z", lai, {**lai_units, ln: "leaf area index (cumulative)"}),
                "dlai": ("zm", dlai, {**lai_units, ln: "leaf area index in layer"}),
                **abs_data_vars,
            },
            attrs={
                "info": info,
                "scheme_name": scheme_name,
                "scheme_long_name": scheme_lname,
                "scheme_short_name": scheme_sname,
                "sza": sza,
                "psi": psi,
                "mu": mu,
                "G": G,
                "K_b": K_b,
            },
        )
        # TODO: add crt1d version?
        # TODO: many of these attrs should be data_vars even though 0-D

        return dset

    def plot_canopy(self):
        """Plot LAI and LAD profiles."""
        _plot_canopy(self)


def _plot_canopy(m):
    """Plot LAI and LAD profiles.

    LAI on interface levels, LAD on mid levels.

    `m` must be :class:`Model`
    """
    p = m._p
    lai = p["lai"]
    z = p["z"]
    dlai = p["dlai"]
    zm = p["zm"]
    dz = p["dz"]

    # mla = p["mla"]

    if np.allclose(dlai, dlai[0]):
        plot_dlai = False
        ncols = 2
        figsize = (4.2, 3.2)
    else:
        plot_dlai = True
        ncols = 3
        figsize = (6, 3.2)

    figname = f"leaf-profiles"
    fig, axs = plt.subplots(1, ncols, figsize=figsize, sharey=True, num=figname)

    fmt = ".-"

    ax1 = axs[0]

    ax1.plot(lai, z, fmt)
    ax1.set_title("cumulative LAI")
    ax1.set_ylabel("height in canopy (m)")

    if plot_dlai:
        ax2 = axs[1]
        ax2.plot(dlai, zm, fmt)
        ax2.set_title("LAI in layer")
        ax2.ticklabel_format(useOffset=False)
        ax3 = axs[2]
    else:
        ax3 = axs[1]

    ax3.plot(dlai / dz, zm, fmt)
    ax3.set_title("LAD")

    for ax in axs:
        ax.grid(True)

    fig.tight_layout()


def _plot_spectral(m):
    """Plot the spectral leaf and soil properties."""
    return NotImplementedError


def _plot_band(dsets, bn):
    """Multi-panel plot of profiles for specified string bandname `bn`:
    'PAR', 'solar', etc.
    """
    if not isinstance(dsets, list):
        print("dsets must be provided as list")
        return

    varnames = [
        [f"aI_{bn}", f"aI_sl_{bn}", f"aI_sh_{bn}"],
        [f"I_dr_{bn}", f"I_df_d_{bn}", f"I_df_u_{bn}"],
    ]  # rows, cols

    nrows = len(varnames)
    ncols = len(varnames[0])

    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(ncols * 2.4, nrows * 3))

    vns = [vn for row in varnames for vn in row]
    for i, vn in enumerate(vns):
        ax = axs.flat[i]
        for dset in dsets:
            da = dset[vn]
            y = da.dims[0]
            da.plot(y=y, ax=ax, label=dset.attrs["scheme_long_name"], marker=".")

    for ax in axs.flat:
        ax.grid(True)

    legfs = 9
    # axs.flat[-1].legend()
    # axs.flat[0].legend()

    h, _ = axs.flat[0].get_legend_handles_labels()
    # fig.legend(handles=h, loc='right')
    # fig.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 0.))
    # fig.legend(handles=h, loc='center', bbox_to_anchor=(1.0, 0.9))
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.1, 0.13), fontsize=legfs)

    fig.tight_layout()


def plot_PAR(dsets=[]):
    """Plot PAR comparison for dsets."""
    _plot_band(dsets, "PAR")


def plot_solar(dsets=[]):
    """Plot spectrally integrated solar comparison for dsets."""
    _plot_band(dsets, "solar")


# def plot_E_closure_spectra():


def create_E_closure_table(dsets=[]):
    """
    For `dsets`, assess energy balance closure by comparing
    computed canopy and soil absorption to incoming minus outgoing
    radiation at the top of the canopy.

    Parameters
    ----------
    dsets : list(xarray.Dataset)
        computed using :func:`to_xr`

    """
    IDs = [ds.attrs["scheme_id"] for ds in dsets]
    columns = [
        "incoming",
        "outgoing (reflected)",
        "soil absorbed",
        "layerwise abs sum",
        "in-out-soil",
        "canopy abs",
    ]
    df = pd.DataFrame(index=IDs, columns=columns)

    for i, ID in enumerate(IDs):
        ds = dsets[i]
        incoming = ds["I_d_solar"][-1].values
        outgoing = ds["I_df_u_solar"][-1].values
        transmitted = ds["I_d_solar"][0].values  # direct+diffuse
        soil_refl = ds["I_df_u_solar"][0].values
        soil_abs = transmitted - soil_refl
        layer_abs_sum = ds["aI_solar"].sum().values
        canopy_abs = (
            ds["I_df_d_solar"][-1].values
            - outgoing
            + ds["I_dr_solar"][-1].values
            - ds["I_dr_solar"][0].values
            + -(ds["I_df_d_solar"][0].values - soil_refl)
        )  # soil abs is down-up diffuse at last layer?
        df.loc[ID, columns[0]] = incoming
        df.loc[ID, columns[1]] = outgoing
        df.loc[ID, columns[2]] = soil_abs
        df.loc[ID, columns[3]] = layer_abs_sum
        df.loc[ID, columns[4]] = incoming - outgoing - soil_abs
        df.loc[ID, columns[5]] = canopy_abs

    return df


def run_cases(m0, cases):
    """For model m, run multiple cases.
    Vary one or more attributes and create combined dataset.

    m0 : model object
        base case to branch off of

    cases : list of dicts ?
        one dict for each case
        keys to change and values to change them to
    """
    case0 = cases[0]
    nparam = len(case0)  # number of things we are varying

    raise NotImplementedError
