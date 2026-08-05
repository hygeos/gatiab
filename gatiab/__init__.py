"""
Gatiab
======

A python package to compute Gaseous Absorption Transmissions at
Instrument Averaged Bands.

Provides
  1. Generation of gas optical depth look-up tables for a given
     AFGL atmosphere, based on the CKDMIP shortwave idealized
     spectra
  2. Computation of the gaseous transmissions at instrument
     averaged bands, as function of the gas content, the airmass
     and the ground pressure, using the spectral response
     functions of the instrument
  3. Rescaling of an optical depth look-up table to a new
     columnar gas content

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with
the code, and the README of `the gatiab repository
<https://github.com/hygeos/gatiab>`_.

Code snippets are indicated by three greater-than signs::

    >>> import gatiab
    >>> gt = gatiab.Gatiab(od_lut)
    >>> trans = gt.calc(gas_content, air_mass, p0, srf_wvl, rsrf)

Use the built-in ``help`` function to view a function's
docstring::

    >>> help(gatiab.ckdmip2od)
"""

from gatiab.constants import (
    ACCEL_DUE_TO_GRAVITY,
    MOLAR_MASS,
    MOLAR_MASS_AIR,
    VERSION,
)
from gatiab.gatiab import (
    Gatiab,
    ckdmip2od,
    get_bands,
    get_binary_mat,
    get_zatm,
    vec_float_indexing,
)

__all__ = [
    'ACCEL_DUE_TO_GRAVITY',
    'MOLAR_MASS',
    'MOLAR_MASS_AIR',
    'VERSION',
    'Gatiab',
    'get_binary_mat',
    'vec_float_indexing',
    'get_zatm',
    'get_bands',
    'ckdmip2od',
]
