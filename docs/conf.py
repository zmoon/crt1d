# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
from pkg_resources import get_distribution  # noreorder

import crt1d


# -- Project information -----------------------------------------------------

project = "crt1d"
author = "Z. Moon"
copyright = f"2020\u2013{datetime.datetime.now().year}, {author}"

# Get version from metadata
release = get_distribution("crt1d").version
version = ".".join(release.split(".")[:2])

# Create some docs content
crt1d.variables._write_params_docs_snippets()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
    "myst_nb",
]


# -- Extension settings ------------------------------------------------------

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}

# napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "xr.Dataset": "xarray.Dataset",
    "xr.DataArray": "xarray.DataArray",
    # NumPy
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "np.ndarray": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
    # pandas
    "pd.DataFrame": "pandas.DataFrame",
}

# include __init__() docstring content in autodocs for classes
# autoclass_content = "both"  # doesn't work with autoapi

# autoapi
autoapi_type = "python"
autoapi_dirs = ["../crt1d/"]
autoapi_add_toctree_entry = False
autoapi_root = "api"  # default: 'autoapi'
autoapi_options = [
    "members",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"  # include __init__ docstring as well as class
autoapi_member_order = "groupwise"  # default is 'bysource'

# bibtex
bibtex_bibfiles = ["crt1d-refs.bib"]  # required in sphinxcontrib-bibtex v2

# autosectionlabel
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "conf.py", "Thumbs.db", ".DS_Store", "../crt1d/*"]


# -- Options for HTML output -------------------------------------------------

# html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"
html_theme = "furo"

html_title = "crt1d"  # shown in top left, overriding "{project} {release} documentation"
html_last_updated_fmt = "%Y-%m-%d"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
