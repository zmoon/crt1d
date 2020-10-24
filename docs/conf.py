# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# -- Project information -----------------------------------------------------

project = "crt1d"
copyright = "2020, Zachary Moon"
author = "Zachary Moon"

# get version from metadata
from pkg_resources import get_distribution

release = get_distribution("crt1d").version
version = ".".join(release.split(".")[:2])

# create some docs info
import crt1d

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
    # "sphinx_rtd_theme",
    # "nbsphinx",
    "sphinxcontrib.bibtex",
    # "myst_parser",  # automatically used to parse .md files
    "autoapi.extension",
    "myst_nb",
]

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}

# aliases
napoleon_type_aliases = {
    "xr.Dataset": ":class:`xarray.Dataset`",
}

# Make nbsphinx detect Jupytext files
nbsphinx_custom_formats = {
    # ".py": ["jupytext.reads", {"fmt": "py:percent"}],
    # ".md": ["jupytext.reads", {"fmt": "md:myst"}],
}
# Figure quality
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    # "--InlineBackend.rc={'figure.dpi': 96}",
]


# include __init__() docstring content in autodocs for classes
autoclass_content = "both"

# autosummary stub generation
# autosummary_generate = True
autodoc_default_flags = ["members"]

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
autoapi_python_class_content = "both"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "conf.py", "Thumbs.db", ".DS_Store", "../crt1d/*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
