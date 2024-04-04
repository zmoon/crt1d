"""
Variable metadata from the ``variables.yml`` file.
"""

from .utils import cf_units_to_tex


class VmdEntry:  # TODO: base on NamedTuple or dataclass??
    """Variable metadata for one variable."""

    def __init__(self, name, params, param_defaults):
        r"""
        Parameters
        ----------
        name : str
            Code variable name associated with the variable.
            Used as the ``name`` for :class:`xarray.DataArray`\s.
        params : dict
            Parameters for this variable.
        param_defaults : dict
            Default parameters (to use if `params` is missing any of the needed).
        """
        self.name = name

        # required (we want it to fail if not provided)
        self.desc = params["desc"]

        # ones that have defaults
        self.s_type = params.get("type", param_defaults["type"])
        self.long_name = params.get("ln", param_defaults["ln"])
        self.intent = params.get("intent", param_defaults["intent"])
        self.is_param = params.get("param", param_defaults["param"])
        self.s_units = str(params.get("units", param_defaults["units"]))
        self.s_units_long = params.get("units_long", param_defaults["units_long"])

        # shape and dims only apply to array_like type
        if self.s_type == "array_like":
            self.s_shape = params["shape"]
            self.dims = _dims_from_s_shape(self.s_shape)
        else:
            self.s_shape = ""
            self.dims = ()

    def da_attrs(self):
        """Return dict of attributes to use when creating an :class:`xarray.DataArray`
        for this variable.
        """
        attrs = {
            "long_name": self.long_name,
            "units": self.s_units,
        }
        if self.s_units_long:
            attrs.update(
                {
                    "units_long": self.s_units_long,
                }
            )

        return attrs

    def dv_tuple(self, data):
        """Construct an :class:`xarray.Dataset` ``data_vars`` tuple."""
        return (self.dims, data, self.da_attrs())

    def param_entry(self, optional=False) -> str:
        """Construct an un-indented NumPy docstring Parameters entry."""
        name = self.name
        s_type = self.s_type
        s_optional = ", optional" if optional else ""
        s_shape = f" *shape*: ``{self.s_shape}``." if self.s_shape else ""
        return f"""
{name}: {s_type}{s_optional}
    {self.long_name}.{s_shape}
        """.strip()

    def list_table_entry(self, fields) -> str:
        """Construct a MyST list-table entry for the docs variable summary table.

        Parameters
        ----------
        fields : list of str
            Parameters to include (in desired order).
        """
        lines = []
        for i, field in enumerate(fields):
            # line prefix
            pre = "* - " if i == 0 else "  - "

            # original attr
            p = getattr(self, field)

            # convert some
            if field == "s_units":
                p = cf_units_to_tex(p) if p else ""
            elif field == "desc":
                p = _desc_for_list_table(p)
            elif field == "long_name":
                p = p or ""
            elif field == "name":
                # hash_link = f"#{p.lower().replace('_', '-')}"  # or could use URL tools
                hash_link = f"#{p.lower()}"
                p = f"[``{p}``]({hash_link})"
            elif field == "s_shape":
                p = _math_shape_from_s_shape(p)

            lines.append(f"{pre}{p}")

        return "\n".join(lines).rstrip()

    def details_sec(self, *, heading_level=3) -> str:
        """Construct details section for docs variables page."""
        pre0 = "#" * heading_level
        header = f"{pre0} ``{self.name}``"
        l_units_long = (
            f"\n* detailed units: {cf_units_to_tex(self.s_units_long)}"
            if self.s_units_long is not None
            else ""
        )
        shape = f"``{self.s_shape}``" if self.s_shape else f"``{self.dims}``"
        return f"""
{header}

{self.desc}

Attributes:
* ``long_name``: {self.long_name}
* units: {cf_units_to_tex(self.s_units)}
{l_units_long}
* type: ``{self.s_type}``
* shape: {shape}
* dims: ``{self.dims}``
        """.strip()

    def __repr__(self):
        return f"{__class__.__name__}(name={self.name}, ...)"

    def __str__(self):
        # fuller representation
        attrs = [
            "s_type",
            "long_name",
            "s_units",
            "s_units_long",
            "s_shape",
            "dims",
            "intent",
            "is_param",
        ]
        s0 = f"{self.name}\n"
        s = "\n".join(f"  {attr}: {getattr(self, attr)!r}" for attr in attrs)
        s += "\n  desc: ..."
        return s0 + s


class Vmd:
    """Container for variable metadata of multiple variables."""

    def __init__(self, vmdes):
        """
        Parameters
        ----------
        vmdes : list of VmdEntry
        """
        self.variables = {vmde.name: vmde for vmde in vmdes}

    def intent(self, intent="in"):
        """Return filtered set of variables that have the specified `intent`.

        Parameters
        ----------
        intent : str, {'in', 'out', 'none'}

        Returns
        -------
        dict
            ``name: VmdEntry``
        """
        if intent is None or intent == "all":
            return self.variables.copy()
        else:
            return {name: vmde for name, vmde in self.variables.items() if vmde.intent == intent}

    def __getitem__(self, name):
        return self.variables[name]

    def __repr__(self):
        s_vmdes = ", ".join(self.variables.keys())
        return f"{__class__.__name__}({s_vmdes})"


def _dims_from_s_shape(s_shape):
    """Detect xarray dims tuple from shape string.
    Helper for `VmdEntry` initialization."""
    assert s_shape[0] == "(" and s_shape[-1] == ")"
    shape_parts = [s for s in s_shape[1:-1].split(",") if s]

    dims = []
    for shape_part_raw in shape_parts:
        shape_part = shape_part_raw.strip()
        assert shape_part[:2] == "n_"

        # special treatment for things off the main grid
        if shape_part[2:] == "z-1":
            dims.append("zm")  # layer midpts
        elif shape_part[2:] == "wl+1":
            dims.append("wle")  # wavelength band edges
        else:
            dims.append(shape_part[2:])

    return dims  # tuple(dims)


def _math_shape_from_s_shape(s_shape):
    """Fancy shape repr for summary table."""
    if s_shape:
        s_shape = s_shape.replace("wl", r"\lambda")
        return f"{{math}}`{s_shape}`"  # myst format
    else:
        return ""


# TODO: def _s_shape_to_tex(s_shape):
#     """Create TeX form of shape specification for docs."""
#     assert s_shape[0] == "(" and s_shape[-1] == ")"
#     shape_parts = s_shape[1:-1].split(",")


def _desc_for_list_table(desc):
    """Properly format the description for MyST list table."""
    lines = [line for line in desc.splitlines() if line.strip()]
    s_parts = []
    for i, line in enumerate(lines):
        pre = "" if i == 0 else " " * 4
        s_parts.append(f"{pre}{line}\n")

    return "\n".join(s_parts)  # ensure blank space between original lines


def _vmd_from_yaml():
    """Load the variable info from the yml file."""
    from pathlib import Path

    import yaml

    p = Path(__file__).parent / "variables.yml"
    with open(p, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    params_allowed = data["variable_params"]
    param_defaults = data["defaults"]
    variables = data["variables"]

    if any(k not in params_allowed for k in param_defaults):
        raise Exception("a param is listed as a default but not allowed")

    vmdes = []
    for name, params in variables.items():
        if any(k not in params_allowed for k in params):
            raise Exception(
                f"param(s) in `{name}` not allowed: "
                f"{', '.join(f'`{k}`' for k in set(params)-set(params_allowed))}"
            )

        vmdes.append(VmdEntry(name, params, param_defaults))

    vmd = Vmd(vmdes)

    return vmd


# Create the Vmd instance
VMD = _vmd_from_yaml()
"""
:class:`Vmd` instance with all the variables from ``variables.yml``.
"""


def _tup(name, data):
    """Shortcut function for creating an :class:`xarray.Dataset` ``data_vars`` tuple
    for variable `name` using the values of `data`
    and the standard variable metadata :const:`VMD`.
    """
    return VMD[name].dv_tuple(data)


def _wl_coord_dict(wl, *, units="μm"):
    return {"wl": (("wl"), wl, {"long_name": "Wavelength", "units": units})}


def _params_list_table(vmdes=None):
    """Form an entire MyST list-table.

    Parameters
    ----------
    vmdes : list(VmdEntry)
        Default is to use all of the known.
    """
    if vmdes is None:
        vmdes = sorted(VMD.variables.values(), key=lambda x: x.name.lower())

    fields = ["name", "s_units", "s_shape", "long_name"]
    entries = "\n".join(vmde.list_table_entry(fields) for vmde in vmdes)

    return f"""
% this table is auto-generated; don't edit directly
```{{list-table}} Summary table
   :widths: 20 20 20 70
   :header-rows: 1

* - name
  - units
  - shape
  - long_name
{entries}
```
    """.rstrip()


def _params_details(vmdes=None):
    if vmdes is None:
        vmdes = sorted(VMD.variables.values(), key=lambda x: x.name.lower())

    entries = "\n\n".join(vmde.details_sec() for vmde in vmdes)

    return entries


# python -c 'import crt1d; crt1d.variables._write_params_docs_snippets()'
def _write_params_docs_snippets():
    import os
    from pathlib import Path

    rtd = os.environ.get("READTHEDOCS", None)
    if rtd is not None:  # on RTD, non-editable package install
        docs = Path(".")
    else:  # dev, editable install
        docs = Path(__file__).parent / "../docs"

    p = docs / "_variables_summary_table_snippet.txt"
    with open(p, "wb") as f:
        f.write(_params_list_table().encode("utf-8"))

    p = docs / "_variables_details_snippet.txt"
    with open(p, "wb") as f:
        f.write(_params_details().encode("utf-8"))


_variables_tpl = """
# Variables

The summary table and detailed descriptions are auto-generated
using the data from ``variables.yml`` and module {{mod}}`crt1d.variables`.

## Summary

Click on a "name" to jump to the details.

{summary_table:s}

## Detailed

Detailed descriptions.

{details:s}
"""


def _write_variables_doc():
    from pathlib import Path

    p = Path(__file__).parent / "../docs" / "variables.md"
    with open(p, "wb") as f:
        f.write(
            _variables_tpl.format(
                summary_table=_params_list_table(),
                details=_params_details(),
            ).encode("utf-8")
        )


# hack module docstring
# include all params
# __doc__ %= {
# "param_in": "\n".join(_param_entry(v) for v in _vmd["in"]),
# "param_out": "\n".join(_param_entry(v) for v in _vmd["out"]),
# "param_table": _all_params_list_table(_vmd["in"]),
# }
