"""Gaseous Absorption Transmissions at Instrument Averaged Bands."""

from gatiab.constants import (
    ACCEL_DUE_TO_GRAVITY,
    MOLAR_MASS,
    MOLAR_MASS_AIR,
    VERSION,
)
from gatiab.gatiab import get_binary_mat
from gatiab.gatiab import vec_float_indexing
from gatiab.gatiab import get_zatm
from gatiab.gatiab import get_bands
from gatiab.gatiab import ckdmip2od
from gatiab.gatiab import Gatiab

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
