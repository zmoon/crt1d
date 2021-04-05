"""
This module contains the model class :class:`Model`, which can be used to
conveniently solve CRT problems using different solvers with minimal boilerplate code needed.
(At least that is the goal.)

This module also contains functions that operate on the model state,
most of which are also attached as class methods.
Functions that operate on :class:`xr.Dataset` s
created by :meth:`Model.to_xr` are in :mod:`.diagnostics`.
"""
# from dataclasses import dataclass
import warnings
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .cases import load_default_case
from .solvers import AVAILABLE_SCHEMES
from .solvers import RET_KEYS_ALL_SCHEMES  # the ones all schemes must return
from .variables import VMD

__all__ = ("Model", "run_sensitivity")


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
    """A general class for testing 1-D canopy radiative transfer schemes."""

    required_input_keys = tuple(
        [k for k in CANOPY_DESCRIPTION_KEYS if k not in ("dlai", "lai_tot", "lai_eff")]
        + [
            "I_dr0_all",
            "I_df0_all",  # spectral (W/m^2/um)
            "wl",
            "dwl",  # for the toc spectra
            "psi",
        ]
    )
    """Required model inputs."""

    _schemes = AVAILABLE_SCHEMES

    vmd = VMD

    def __init__(
        self,
        scheme="2s",
        nlayers=60,
        **p_kwargs,
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
        **p_kwargs
            Model parameter keyword arguments passed on to :meth:`update_p()`.
        """
        # load default case, for given nlayers
        self.nlayers = nlayers
        self.p_default = load_default_case(nlayers=self.nlayers)
        """Default parameter settings dict."""

        # base initial settings on default
        self._p = deepcopy(self.p_default)

        # assign scheme
        self.assign_scheme(scheme)  # assigns scheme info dict to self.scheme

        # add parameters in?
        if p_kwargs:
            self.update_p(**p_kwargs)
        else:
            # check inputs (included in update_p)
            self._check_inputs()

        # run/output variables
        self._run_count = 0  # TODO: store last_state?

        self.absorption = None  # initially no absorption data
        """Absorption outputs calculated from the standard outputs :attr:`out`."""

        self.out = {}
        """Scheme standard outputs."""

        self.out_extra = {}
        """Scheme extra outputs, such as absorption. Only some schemes provide any."""

    @property
    def p(self):
        """Model parameters are indended to be read only (updated with :meth:`update_p`).
        Invoking this just prints a message.
        """
        print(
            "Please update parameters using `.update_p()`! Changes to `.p` will not be stored!\n"
            "Extract (copy) the parameters using `.copy_p()` or summarize using `.print_p()`."
        )

    def print_p(self):
        """Pretty print the parameters."""
        import pprint

        pp = pprint.PrettyPrinter(indent=1)
        with np.printoptions(precision=3, threshold=7):
            pp.pprint(self._p)

    def copy_p(self):
        """Return a copy of the parameters dict."""
        return deepcopy(self._p)

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
        """Using the :const:`~crt1d.solvers.AVAILABLE_SCHEMES` dict,
        assign scheme and necessary scheme attrs.
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
        """Update parameters, if the validation passes.

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

    def update_spectra(self, ds):
        """Update irradiance and leaf/soil optical property spectra
        from :class:`xarray.Dataset` `ds`.
        """
        self.update_p(
            I_dr0_all=ds["I_dr"].values,
            I_df0_all=ds["I_df"].values,
            wl=ds["wl"].values,
            dwl=ds["dwl"].values,
            wle=ds["wle"].values,
            leaf_t=ds["tl"].values,
            leaf_r=ds["rl"].values,
            soil_r=ds["rs"].values,
            wl_leafsoil=ds["wl"].values,
        )
        return self

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
        # assert dlai.sum() == lai[0]
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
        # check wavelengths are compatible etc.
        # should already have been done at least once by now, but whatever
        self._check_inputs()

        # construct dict of kwargs to pass to the solver
        scheme = self.scheme
        p = self._p
        args = {k: p[k] for k in scheme["args"]}

        # run
        sol = scheme["solver"](**args, **extra_solver_kwargs)

        # use the dict returned by the solver to update our state
        self.out.update({k: v for k, v in sol.items() if k in RET_KEYS_ALL_SCHEMES})
        self.out_extra.update(
            {f"{k}_scheme": v for k, v in sol.items() if k not in RET_KEYS_ALL_SCHEMES}
        )

        self._run_count += 1

        return self  # for chaining

    @property
    def out_all(self):
        """Standard and extra outputs."""
        return {**self.out, **self.out_extra}

    def calc_absorption(self):
        """Calculate layerwise absorption variables."""
        if self._run_count == 0:
            raise Exception("Must run the model first.")

        absorption = _calc_absorption(self)
        # update model attr
        self.absorption = absorption

        return self  # for chaining

    def to_xr(self, *, info=""):
        """Construct and return an :class:`xarray.Dataset`.

        Parameters
        ----------
        info : str
            Extra information about the run/model to be stored in the dataset.
        """
        # import datetime
        import crt1d

        if self._run_count == 0:
            raise Exception("Must run the model before creating the dataset.")
        p = self._p
        out = self.out_all
        out_extra = self.out_extra  # non-standard outputs, such as absorption
        #
        # -- canopy descrip
        lai = p["lai"]
        dlai = p["dlai"]
        z = p["z"]  # at lai values (layer interfaces)
        zm = p["zm"]
        #
        # -- wavelength grid
        wl = p["wl"]
        dwl = p["dwl"]
        #
        # -- radiation geometry/setup
        psi = p["psi"]
        sza = np.rad2deg(psi)
        G = p["G"]
        K_b = p["K_b"]
        #
        # -- scheme info
        scheme_lname = self.scheme["long_name"]
        scheme_sname = self.scheme["short_name"]
        scheme_name = self.scheme["name"]
        #
        # -- canopy RT solution
        Idr = out["I_dr"]
        Idfd = out["I_df_d"]
        Idfu = out["I_df_u"]
        F = out["F"]

        # -- define fn to look up variable metadata and create data_vars tuple
        def tup(name, data):
            return self.vmd[name].dv_tuple(data)

        # -- construct data vars for absorption (many)
        abs_data_vars = {}

        # -- standard absorption calculations (layer in-out)
        #    we can use the standard metadata
        abs_post = self.absorption if self.absorption is not None else {}
        abs_data_vars.update({k: tup(k, v) for k, v in abs_post.items()})

        # -- scheme's absorption
        abs_scheme = {f"{k}": v for k, v in out_extra.items() if k[:2] == "aI"}
        for name, arr in abs_scheme.items():
            n_z = arr.shape[0]
            if n_z == z.size:  # some schemes provide absorption on interface levels
                dims = ("z", "wl")
            elif n_z == zm.size:
                dims = ("zm", "wl")
            else:
                raise ValueError("Scheme absorption output has too many or too few levels.")

            name0 = name[:-7]  # without the `_scheme` suffix
            try:
                attrs = self.vmd[name0].da_attrs()
            except KeyError:
                raise Exception(f"Scheme absorbance variable {name0} not found in vmd.")

            abs_data_vars[name] = (dims, arr, attrs)

        dset = xr.Dataset(
            coords={
                "z": tup("z", z),
                "wl": tup("wl", wl),
                "zm": tup("zm", zm),
            },
            data_vars={
                "I_dr": tup("I_dr", Idr),
                "I_df_d": tup("I_df_d", Idfd),
                "I_df_u": tup("I_df_u", Idfu),
                "F": tup("F", F),
                "I_d": tup("I_d", Idr + Idfd),
                "dwl": tup("dwl", dwl),
                "lai": tup("lai", lai),
                "dlai": tup("dlai", dlai),
                #
                **abs_data_vars,
                #
                "psi": tup("psi", psi),
                "sza": tup("sza", sza),
                "G": tup("G", G),
                "K_b": tup("K_b", K_b),
            },
            attrs={
                "info": info,
                "scheme_name": scheme_name,
                "scheme_long_name": scheme_lname,
                "scheme_short_name": scheme_sname,
                "crt1d_version": crt1d.__version__,
                # "run_date": datetime.datetime.now(),
            },
        )
        return dset

    def plot_canopy(self):
        """Plot LAI and LAD profiles."""
        _plot_canopy(self)

    def plot_toc_spectra(self):
        """Plot the toc (top-of-canopy) irradiance spectra used for the upper boundary condition."""
        _plot_toc_spectra(self)

    def plot_leafsoil_spectra(self):
        """Plot the leaf and soil optical properties spectra used by the model."""
        _plot_leafsoil_spectra(self)


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

    if np.allclose(dlai, dlai[0]):
        plot_dlai = False
        ncols = 2
        figsize = (4.2, 3.2)
    else:
        plot_dlai = True
        ncols = 3
        figsize = (6, 3.2)

    figname = "leaf-profiles"
    fig, axs = plt.subplots(1, ncols, figsize=figsize, sharey=True, num=figname)

    fmt = ".-"

    ax1 = axs[0]

    ax1.plot(lai, z, fmt)
    ax1.set_title("Cumulative LAI")
    ax1.set_ylabel("Height in canopy (m)")

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


def _plot_toc_spectra(m):
    """Plot the toc irradiance spectra."""
    p = m._p
    dwl = p["dwl"]
    wl = p["wl"]
    Idr = p["I_dr0_all"]
    Idf = p["I_df0_all"]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(wl, Idr / dwl, label="direct")
    ax.plot(wl, Idf / dwl, label="diffuse")
    ax.plot(wl, (Idr + Idf) / dwl, label="total")

    ax.set(xlabel="Wavelength (μm)", ylabel="Spectral irradiance (W m$^{-2}$ μm$^{-1}$)")

    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_ylim(ymin=0)

    ax.legend()
    fig.tight_layout()


def _plot_leafsoil_spectra(m):
    """Plot the spectral leaf and soil properties."""
    p = m._p
    wl = p["wl_leafsoil"]
    rl = p["leaf_r"]
    tl = p["leaf_t"]
    rs = p["soil_r"]

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 5), sharex=True, sharey=True)

    # leaf
    l1 = rl
    l2 = 1 - tl
    ax1.plot(wl, l1, label="Reflectance", lw=2.0)
    ax1.plot(wl, l2, label="Transmittance\n(from top x-axis)", lw=2.0)
    ax1.fill_between(wl, l1, l2, color="0.70", alpha=0.5)
    ax1.legend()
    ax1.set_ylim((0, 1))
    ax1.set_title("Grey = Absorbance", loc="left", color="0.4", fontsize=9)
    # ^ indicate what the grey means
    ax1.text(0.01, 0.97, "Leaf", ha="left", va="top", transform=ax1.transAxes)

    # soil
    l1 = rs
    l2 = np.ones_like(l1)
    ax2.plot(wl, rs, label="Reflectance (soil albedo)")
    ax2.fill_between(wl, l1, l2, color="0.70", alpha=0.5)
    ax2.text(0.01, 0.97, "Soil", ha="left", va="top", transform=ax2.transAxes)

    ax2.set(xlabel="Wavelength (μm)", ylabel="")
    ax2.autoscale(enable=True, axis="x", tight=True)
    ax2.set_ylim(ymin=0)

    fig.tight_layout()


def _calc_absorption(m):
    """Calculate layerwise absorption using the light profiles etc."""
    p = m._p
    out = m.out

    lai = p["lai"]
    dlai = p["dlai"]
    # G = p["G"]  # fractional leaf area projected in direction psi
    K_b = p["K_b"]  # G/cos(psi)

    K_b = 0.577350269  # testing Bonan

    leaf_r = p["leaf_r"]
    leaf_t = p["leaf_t"]
    leaf_a = 1 - (leaf_r + leaf_t)  # leaf element absorption coeff

    # wl = p["wl"]
    I_dr = out["I_dr"]
    I_df_d = out["I_df_d"]
    I_df_u = out["I_df_u"]
    # I_d = I_dr + I_df_d  # direct+diffuse downward irradiance

    # TODO: include clump factor in f_sl and absorption calculations

    # Sunlit leaf fraction
    # f_sl_interfaces = np.exp(-K_b * lai)  # at interface levels
    # f_sl = f_sl_interfaces[:-1] + 0.5 * np.diff(f_sl_interfaces)  # at mid levels (abs.)
    # TODO: better to use LAI midpts, not f_sl_interfaces midpts
    # TODO: and `zm` then should be the corresponding z values from interp, not `z` midpts
    laim = (lai[:-1] + lai[1:]) / 2
    f_sl = np.exp(-K_b * laim)
    f_sh = 1 - f_sl

    # Compute total layerwise absorbed (by plant, but not per unit LAI)
    nlev = lai.size
    i = np.arange(nlev - 1)
    ip1 = i + 1
    a = I_dr[ip1] - I_dr[i] + I_df_d[ip1] - I_df_d[i] + I_df_u[i] - I_df_u[ip1]
    # ^ a: layerwise irradiance absorption in all bands
    #      as inputs - outputs
    #      actual W/m2 absorption, not per unit LAI

    # Absorbed direct depends on the sunlit leaf fraction
    # I_dr0 = I_dr[-1, :][np.newaxis, :]
    # a_dr =  I_dr0 * (K_b*f_sl*dlai)[:,np.newaxis] * leaf_a
    # a_dr = I_dr0 * (1 - np.exp(-K_b * f_sl * dlai))[:, np.newaxis] * leaf_a
    a_dr = (
        I_dr[1:, :]  # direct beam penetration above level
        * (1 - np.exp(-K_b * dlai))[:, np.newaxis]  # 1 - tau_b (transmittance through layer)
        * leaf_a  # frac absorbed (as opposed to scattered)
    )
    # ^ direct beam absorption (sunlit leaves only by definition)
    #
    # **technically should be computed with exp**
    #   1 - exp(-K_b*L)
    # K_b*L is an approximation for small L
    # e.g.,
    #   K=1, L=0.01, 1-exp(-K*L)=0.00995

    # Absorbed diffuse is the remaining fraction of absorbed radiation
    a_df = a - a_dr

    # Sunlit (sl) and shaded (sh) leaves
    a_df_sl = a_df * f_sl[:, np.newaxis]
    a_df_sh = a_df * f_sh[:, np.newaxis]
    a_sl = a_df_sl + a_dr
    a_sh = a_df_sh
    assert np.allclose(a_sl + a_sh, a)

    return {
        "aI": a,
        "aI_df": a_df,
        "aI_dr": a_dr,
        "aI_sh": a_sh,
        "aI_sl": a_sl,
        "aI_df_sl": a_df_sl,
        "aI_df_sh": a_df_sh,
        # "I_d": I_d,
        # "I_dr": I_dr,
        # "I_df_d": I_df_d,
        # "I_df_u": I_df_u,
    }


def run_sensitivity(m0, p_sets):
    """For model m, run multiple cases.
    Vary one or more attributes and create combined dataset.

    m0 : model object
        base case to branch off of
    p_sets : dict
        keys: param to change; values: list of values for the param

    Returns
    -------
    xr.Dataset
        With a new dimension for each key in `p_sets`
    """
    raise NotImplementedError
