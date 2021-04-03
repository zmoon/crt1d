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

Input data
----------
.. autosummary::

   crt1d.cases
   crt1d.data
   crt1d.spectra

Leaf arrangement
----------------
.. autosummary::

   crt1d.leaf_angle
   crt1d.leaf_area


Other
=====

Solvers
-------

:mod:`~crt1d.solvers` is mostly private, not intended to be used directly. Instead, the solvers
are used while attached to a :class:`crt1d.Model` instance.

.. autosummary::

   crt1d.solvers
   crt1d.solvers.common

Internal tools
--------------

.. autosummary::

   crt1d.utils
   crt1d.variables


AutoAPI
=======

.. toctree::
   :maxdepth: 3

   api/crt1d/index
