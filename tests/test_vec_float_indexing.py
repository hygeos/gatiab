#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for vec_float_indexing and get_binary_mat.

vec_float_indexing is a port of the multilinear interpolation
performed by LUT.__getitem__ / Idx from the luts package
(https://github.com/hygeos/luts) and must give the same results.
It is also used outside gatiab (e.g. the atmosphere module of the
smartg_pv package), hence the exhaustive coverage.

These tests are self-contained: no ckdmip/atmosphere data
directories needed. The oracle for random-data cross-checks is
scipy's RegularGridInterpolator on integer axes (verified during
development to match luts to ~1e-16).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from gatiab import vec_float_indexing, get_binary_mat


@pytest.fixture
def m1() -> NDArray:
    # The 3x2 array used in the vec_float_indexing docstring examples
    return np.arange(6).reshape(3, 2)


def rgi_oracle(
    data: NDArray,
    keys_arrays: list[NDArray],
    fill_value: float | None = np.nan,
) -> NDArray:
    """Reference multilinear interpolation of ``data`` at
    fractional indices, one 1-D float array per dimension (no
    slices).
    """
    axes = [np.arange(n) for n in data.shape]
    interp = RegularGridInterpolator(
        axes, data, bounds_error=False, fill_value=fill_value
    )
    return interp(np.stack(keys_arrays, axis=-1))


class TestDocstringExamples:
    # Executable version of the docstring examples (doctest
    # collection is not enabled: the docstring reprs are not
    # numpy>=2 clean)

    def test_array_key_with_slice(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([0.8, 1.1, 1.5]), slice(None)])
        assert_allclose(res, [[1.6, 2.6], [2.2, 3.2], [3.0, 4.0]], rtol=1e-12)

    def test_scalar_keys(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [0.8, 0.0])
        assert_allclose(res, 1.6, rtol=1e-12)


class TestExactGridPoints:
    # Integer-valued float keys must reproduce the grid values exactly
    # (interpolation weights are exactly 0 and 1)

    @pytest.mark.parametrize('key_val', [0.0, 1.0, 2.0])
    def test_integer_valued_array_key(
        self, m1: NDArray, key_val: float
    ) -> None:
        res = vec_float_indexing(m1, [np.array([key_val]), slice(None)])
        assert_array_equal(res, m1[[int(key_val)], :])

    @pytest.mark.parametrize('key_val', [0.0, 1.0, 2.0])
    def test_integer_valued_scalar_key(
        self, m1: NDArray, key_val: float
    ) -> None:
        res = vec_float_indexing(m1, [key_val, 0.0])
        assert res == m1[int(key_val), 0]

    def test_full_identity(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([0.0, 1.0, 2.0]), slice(None)])
        assert_array_equal(res, m1)


class TestEdgeKeys:
    # key == N-1 raised IndexError before commit a098939; key == 0
    # covers the other clamp boundary

    def test_array_key_equals_n_minus_1(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([2.0]), slice(None)])
        assert_array_equal(res, [[4.0, 5.0]])

    def test_array_key_equals_zero(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([0.0]), slice(None)])
        assert_array_equal(res, [[0.0, 1.0]])

    def test_scalar_keys_at_upper_edges(self, m1: NDArray) -> None:
        assert vec_float_indexing(m1, [2.0, 1.0]) == 5.0


class TestSlicePositions:
    # A slice placed before an interpolated key must not shift the
    # corner selection of the following dimensions (regression
    # test: bit position among interpolated keys vs raw dimension
    # position, as in luts)

    def test_leading_slice(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [slice(None), np.array([0.5])])
        assert_allclose(res, [[0.5], [2.5], [4.5]], rtol=1e-12)

    def test_middle_slice(self) -> None:
        data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        res = vec_float_indexing(
            data, [np.array([0.5]), slice(None), np.array([1.5])]
        )
        assert_allclose(res, [[7.5, 11.5, 15.5]], rtol=1e-12)

    def test_h2o_call_site_pattern(self) -> None:
        # ckdmip2od H2O 'fast' path: keys = [P, T, slice(None), h2o_mf]
        # on data of shape (nP, nT, nwc, nmf)
        rng = np.random.default_rng(7)
        data = rng.random((3, 4, 2, 5))
        kp = rng.uniform(0.0, 2.0, 6)
        kt = rng.uniform(0.0, 3.0, 6)
        km = rng.uniform(0.0, 4.0, 6)
        res = vec_float_indexing(data, [kp, kt, slice(None), km])
        assert res.shape == (6, 2)
        for iw in range(data.shape[2]):
            expected = rgi_oracle(data[:, :, iw, :], [kp, kt, km])
            assert_allclose(res[:, iw], expected, rtol=0, atol=1e-14)


class TestScalarKeys:

    def test_all_scalar_keys_return_scalar(self, m1: NDArray) -> None:
        # Crashed with AttributeError ('float' has no 'reshape') before
        # commit a098939
        res = vec_float_indexing(m1, [0.5, 0.5])
        assert np.ndim(res) == 0
        assert_allclose(res, 1.5, rtol=1e-12)

    def test_scalar_key_with_slice(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [0.5, slice(None)])
        assert res.shape == (2,)
        assert_allclose(res, [1.0, 2.0], rtol=1e-12)


class TestOracleCrossCheck:
    # Property-style check on random data against
    # RegularGridInterpolator, all dimensions indexed with
    # in-range 1-D float arrays

    @pytest.mark.parametrize('shape', [(4, 5), (4, 5, 6), (3, 4, 5, 6)])
    def test_matches_regular_grid_interpolator(
        self, shape: tuple[int, ...]
    ) -> None:
        rng = np.random.default_rng(42)
        data = rng.random(shape)
        keys = [rng.uniform(0.0, n - 1, 10) for n in shape]
        res = vec_float_indexing(data, keys)
        assert_allclose(res, rgi_oracle(data, keys), rtol=0, atol=1e-14)


class TestCallSitePatterns:
    # Shapes and key layouts as used by ckdmip2od with
    # float_indexing='fast'

    def test_3d_trailing_slice(self) -> None:
        # Non-H2O gases: keys = [P, T, slice(None)] on (nP, nT, nwc)
        rng = np.random.default_rng(11)
        data = rng.random((3, 4, 5))
        kp = rng.uniform(0.0, 2.0, 6)
        kt = rng.uniform(0.0, 3.0, 6)
        res = vec_float_indexing(data, [kp, kt, slice(None)])
        assert res.shape == (6, 5)
        for iw in range(data.shape[2]):
            expected = rgi_oracle(data[:, :, iw], [kp, kt])
            assert_allclose(res[:, iw], expected, rtol=0, atol=1e-14)

    def test_keys_exactly_on_last_grid_point(self) -> None:
        # interp1d(P_fl, arange(nP)) can return exactly N-1: the whole
        # result row must equal the last grid row (a098939 regression,
        # call-site shape)
        rng = np.random.default_rng(12)
        data = rng.random((3, 4, 5))
        res = vec_float_indexing(
            data, [np.array([2.0, 0.0]), np.array([3.0, 0.0]), slice(None)]
        )
        assert_allclose(res[0], data[2, 3, :], rtol=1e-12)
        assert_allclose(res[1], data[0, 0, :], rtol=1e-12)


class TestDimensionality:

    def test_1d_data_single_key(self) -> None:
        # Raised IndexError when corner bits came from get_binary_mat(1)
        data = np.arange(5, dtype=np.float64)
        res = vec_float_indexing(data, [np.array([1.5, 2.5])])
        assert_allclose(res, [1.5, 2.5], rtol=1e-12)

    def test_5d_data_all_array_keys(self) -> None:
        # Raised IndexError when corner bits came from get_binary_mat(5)
        # (5**2 rows < 2**5 combinations)
        rng = np.random.default_rng(5)
        shape = (3, 3, 3, 3, 3)
        data = rng.random(shape)
        keys = [rng.uniform(0.0, n - 1, 7) for n in shape]
        res = vec_float_indexing(data, keys)
        assert_allclose(res, rgi_oracle(data, keys), rtol=0, atol=1e-14)


class TestExtrapolation:
    # Keys outside [0, N-1] extrapolate linearly from the edge cell
    # (floor index clamped to [0, N-2])

    def test_below_zero(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([-0.5]), slice(None)])
        assert_allclose(res, [[-1.0, 0.0]], rtol=1e-12)

    def test_above_n_minus_1(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([2.5]), slice(None)])
        assert_allclose(res, [[5.0, 6.0]], rtol=1e-12)

    def test_extrapolation_oracle(self) -> None:
        rng = np.random.default_rng(123)
        shape = (4, 5, 6)
        data = rng.random(shape)
        keys = [rng.uniform(-0.9, n - 0.1, 12) for n in shape]
        # fill_value=None: linear extrapolation in
        # RegularGridInterpolator
        res = vec_float_indexing(data, keys)
        assert_allclose(
            res, rgi_oracle(data, keys, fill_value=None), rtol=0, atol=1e-14
        )


class TestInputsAndDtypes:

    def test_keys_list_not_mutated(self, m1: NDArray) -> None:
        k0 = np.array([0.8, 1.1])
        k1 = slice(None)
        keys = [k0, k1]
        k0_copy = k0.copy()
        vec_float_indexing(m1, keys)
        assert keys[0] is k0
        assert keys[1] is k1
        assert_array_equal(k0, k0_copy)

    def test_int_data_returns_float64(self, m1: NDArray) -> None:
        res = vec_float_indexing(m1, [np.array([0.5]), slice(None)])
        assert res.dtype == np.float64

    def test_float32_data_returns_float64(self, m1: NDArray) -> None:
        # Current behaviour (weights are float64), not a contract
        res = vec_float_indexing(
            m1.astype(np.float32), [np.array([0.5]), slice(None)]
        )
        assert res.dtype == np.float64


class TestGetBinaryMat:

    def test_rows_are_little_endian_bits(self) -> None:
        bmat = get_binary_mat(3)
        assert bmat.shape == (9, 3)
        assert bmat.dtype == np.int32
        for i in range(8):
            assert_array_equal(bmat[i], [(i >> b) & 1 for b in range(3)])
