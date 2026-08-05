"""Gaseous Absorption Transmissions at Instrument Averaged Bands."""

from gatiab.gatiab import get_binary_mat
from gatiab.gatiab import vec_float_indexing
from gatiab.gatiab import get_zatm
from gatiab.gatiab import get_bands
from gatiab.gatiab import ckdmip2od
from gatiab.gatiab import Gatiab

__all__ = [
    'get_binary_mat',
    'vec_float_indexing',
    'get_zatm',
    'get_bands',
    'ckdmip2od',
    'Gatiab',
]
