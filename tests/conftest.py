#!/usr/bin/env python

"""Pytest configuration of the gatiab test suite.

The tests of the gatiab module need the CKDMIP idealized look-up
tables and the AFGL atmosphere files, whose directories are given
by the --dir-ckdmip and --dir-atm options. The tests needing them
are skipped when the options are not provided (e.g. on continuous
integration).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --dir-ckdmip and --dir-atm options."""
    parser.addoption(
        "--dir-ckdmip", action="store", default=None,
        help="ckdmip directory path",
    )
    parser.addoption(
        "--dir-atm", action="store", default=None,
        help="atmosphere directory path",
    )
