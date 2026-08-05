#!/usr/bin/env python
"""Package-level constants.

This module defines the gatiab version (read automatically, with
pyproject.toml as single source of truth), the physical constants
and the gas molar masses used by the optical depth and transmission
computations.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    tomllib = None

DIR_ROOT: Path = Path(__file__).resolve().parent.parent


def _get_version() -> str:
    """Read the gatiab version.

    The version is read from pyproject.toml when running from a
    source checkout (python >= 3.11), else from the installed
    package metadata, with '0.0.0' as final fallback.

    Returns
    -------
    str
        The gatiab version
    """
    if tomllib is not None:
        try:
            with open(DIR_ROOT / 'pyproject.toml', 'rb') as f:
                return tomllib.load(f)['project']['version']
        except (FileNotFoundError, KeyError):
            pass
    try:
        return _installed_version('gatiab')
    except PackageNotFoundError:
        return '0.0.0'


VERSION: str = _get_version()

ACCEL_DUE_TO_GRAVITY: float = 9.80665  # m s-2
MOLAR_MASS_AIR: float = 28.970  # g mol-1 dry air

MOLAR_MASS: dict[str, float] = {
    'h2o': 18.0152833,
    'co2': 44.011,
    'o3': 47.9982,
    'n2o': 44.013,
    'co': 28.0101,
    'ch4': 16.043,
    'o2': 31.9988,
    'cfc11': 137.3686,
    'cfc12': 120.914,
    'hcfc22': 86.469,
    'ccl4': 153.823,
    'no2': 46.0055,
    'n2': 28.0134,
}
