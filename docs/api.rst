API
***

Top level
=========

The model  class is included in the top-level namespace for ease-of-use.

.. autosummary::

   crt1d


Public submodules
=================

Analysis
--------
.. autosummary::

   crt1d.diagnostics

Setup
-----
.. autosummary::
   :recursive:

   crt1d.cases
   crt1d.spectra
   crt1d.data


Leaf arrangement
----------------
.. autosummary::

   crt1d.leaf_area
   crt1d.leaf_angle



Other
=====

solvers
-------

:mod:`solvers` is mostly private, not intended to be used directly. Instead, the solvers
are used while attached to a :class:`crt1d.Model` instance.

.. autosummary::

   crt1d.solvers


AutoAPI
=======

.. toctree::
   :maxdepth: 3

   api/crt1d/index
