"""
Some utility functions, mostly for internal use.
"""


def add_snippets(func, snippets):
    """Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names."""
    # credit: https://github.com/lukelbd/proplot/blob/master/proplot/internals/docstring.py
    # but for now not using a global snippets dict
    import inspect

    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= {key: value.strip() for key, value in snippets.items()}
    return func


def cf_units_to_tex(s: str):
    """Convert CF-style units string to TeX-like.
    (In order to get exponents in plot labels, etc.)
    """
    import re

    if s == "1":  # hack for now to allow CF unitless
        return s

    def expify(match):
        m = match.group(0)
        return f"$^{{{m}}}$"

    # this regex matches integers with optional negative sign (hyphen)
    # TODO: more careful match only to the right of a base unit (one with no exp)
    s_new = re.sub(r"-?\d", expify, s)

    return s_new
