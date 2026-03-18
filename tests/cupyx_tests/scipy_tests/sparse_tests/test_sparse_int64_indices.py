from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupyx import cusparse
from cupyx.scipy import sparse


# Index value that exceeds INT32_MAX (= 2**31 - 1), forcing int64.
_LARGE = 2**31 + 1


class TestInt64Construction:
    """Index dtype is chosen by get_index_dtype, not forced to int32.

    The key change: constructors call get_index_dtype(check_contents=True)
    instead of unconditionally casting to int32.  Small values still produce
    int32 (no regression); large values now produce int64.
    """

    def test_csr_large_col_index_uses_int64(self):
        # Column index > INT32_MAX → both indices and indptr use int64.
        # CSR invariant: indices and indptr always share the same dtype.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_csc_large_row_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_coo_large_col_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64
        assert int(m.col[1]) == _LARGE

    def test_csr_int64_with_small_values_stays_int32(self):
        # get_index_dtype(check_contents=True) downcasts int64 arrays when
        # all values fit in int32.  This is the scipy-compatible behavior.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 5], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        assert m.indices.dtype == cupy.int32
        assert m.indptr.dtype == cupy.int32

    def test_coo_int64_with_small_values_stays_int32(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 5], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)))
        assert m.row.dtype == cupy.int32
        assert m.col.dtype == cupy.int32

    def test_empty_csr_large_shape_uses_int64(self):
        # Shape-only construction: max(shape) > INT32_MAX → int64 index arrays,
        # even when there are no stored elements.
        m = sparse.csr_matrix((2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64

    def test_empty_coo_large_shape_uses_int64(self):
        m = sparse.coo_matrix((2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64

    @testing.with_requires('scipy')
    def test_from_scipy_csr_preserves_int64(self):
        # scipy uses its own get_index_dtype; CuPy must trust and preserve it.
        import scipy.sparse
        data = numpy.array([1.0, 2.0])
        indices = numpy.array([0, _LARGE], dtype=numpy.int64)
        indptr = numpy.array([0, 1, 2], dtype=numpy.int64)
        sp = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        m = sparse.csr_matrix(sp)
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE


class TestInt64FormatConversion:
    """Pure-CuPy fallbacks for tocoo / tocsr / tocsc with int64.

    cuSPARSE's xcsr2coo / xcoo2csr / csr2cscEx2 only accept int32 pointers.
    The fallbacks use searchsorted (for indptr expansion) and unique+scatter
    (for indptr construction), avoiding the 2×large-allocation OOM of bincount.
    """

    def _make_int64_csr(self):
        """2-row CSR: row 0 → col 0, row 1 → col _LARGE."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def _make_int64_coo(self):
        """2-entry COO: (row=0, col=0, val=1.0), (row=1, col=_LARGE, val=2.0)."""
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        return sparse.coo_matrix(
            (data, (row, col)), shape=(2, _LARGE + 1))

    def test_csr_tocoo_int64(self):
        # Earlier in development, xcsr2coo read int64 indptr as int32, producing
        # silently wrong row indices (e.g. row=[1,0] instead of [0,1]).
        coo = self._make_int64_csr().tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == 1
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == _LARGE

    def test_coo_tocsr_int64(self):
        csr = self._make_int64_coo().tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert int(csr.indptr[0]) == 0
        assert int(csr.indptr[1]) == 1
        assert int(csr.indptr[2]) == 2
        assert int(csr.indices[1]) == _LARGE

    def test_csc_tocoo_int64(self):
        # csc2coo has its own searchsorted fallback, separate from csr2coo.
        # CSC: col 0 → row 0, col 1 → row _LARGE.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per column
        csc = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        coo = csc.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == _LARGE
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == 1

    def test_csr_tocsc_int32_regression(self):
        # The int32 tocsc path must remain correct as we make changes.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int32)
        csr = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        csc = csr.tocsc()
        assert csc.indices.dtype == cupy.int32
        assert csc.indptr.dtype == cupy.int32
        testing.assert_array_equal(csc.toarray(), csr.toarray())

    @testing.slow
    def test_csr_tocsc_int64(self):
        # _cupy_csr2csc_int64 uses unique+scatter for the CSC indptr.
        # This test requires ~17 GB for the CSC indptr; skipped if OOM.
        n = numpy.iinfo(numpy.int32).max + 3  # INT32_MAX + 3 ≈ 2.15 B
        try:
            data = cupy.array([1.0, 2.0])
            indices = cupy.array([0, n - 1], dtype=cupy.int64)
            indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
            csr = sparse.csr_matrix((data, indices, indptr), shape=(2, n))
            csc = csr.tocsc()
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory')
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        # Row indices in CSC: col 0 holds row 0, col n-1 holds row 1.
        assert int(csc.indices[0]) == 0
        assert int(csc.indices[1]) == 1


class TestInt64Sort:
    """Lexsort-based fallbacks for csrsort, cscsort, sum_duplicates.

    cuSPARSE xcsrsort / xcscsort / xcoosort accept only int32 index pointers.
    """

    def test_csr_sort_indices_int64(self):
        # csrsort int64 path: expand indptr via searchsorted, then lexsort.
        # Row 0 has columns [_LARGE+1, _LARGE] (deliberately unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller col first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0           # data reordered with indices
        assert float(m.data[1]) == 1.0

    def test_csc_sort_indices_int64(self):
        # cscsort int64 path: column 0 has rows [_LARGE+1, _LARGE] (unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 2], dtype=cupy.int64)  # 2 nnz in col 0
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 2, 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller row first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0
        assert float(m.data[1]) == 1.0

    def test_coo_sum_duplicates_int64(self):
        # Previously, the ElementwiseKernel declared 'int32 src_col',
        # silently truncating int64 col values > INT32_MAX to their low 32 bits.
        # Two duplicate entries: both at (row=0, col=_LARGE).
        data = cupy.array([2.0, 3.0])
        row = cupy.array([0, 0], dtype=cupy.int64)
        col = cupy.array([_LARGE, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(1, _LARGE + 1))

        m.sum_duplicates()

        assert m.nnz == 1
        assert m.col.dtype == cupy.int64
        assert int(m.col[0]) == _LARGE  # must not be truncated to int32
        assert float(m.data[0]) == pytest.approx(5.0)

    def test_csr_sort_indices_int32_regression(self):
        # The int32 csrsort path must remain correct.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([2, 0, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, 3))
        m.sort_indices()
        assert m.has_sorted_indices
        testing.assert_array_equal(m.indices, cupy.array([0, 1, 2]))
        testing.assert_array_equal(m.data, cupy.array([2.0, 3.0, 1.0]))


class TestInt64ArithmeticFallback:
    """Sparse addition with int64 indices — pure-CuPy fallback.

    csrgeam2 routes int64 inputs to _cupy_csrgeam_int64 *before* checking
    cuSPARSE availability, so the path works on any CUDA version.

    The fallback: expand indptr→row via searchsorted, concatenate COO entries
    from both matrices, call sum_duplicates() to merge overlapping positions,
    then convert back to CSR.  Index dtype is
    numpy.result_type(a.indices.dtype, b.indices.dtype) throughout.

    Note: cusparseSpGEAM (the Generic API path) is absent from all public
    cuSPARSE releases through 12.7.9, so the pure-CuPy fallback is always
    active on current installations.
    """

    # Shape has _LARGE+2 columns so a column index of _LARGE is valid and
    # forces int64.  Only 2 rows, so indptr has 3 elements (cheap).
    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, col, value=1.0):
        """Single-entry CSR: row 0 has one nonzero at (0, col)."""
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=self._shape)

    def test_add_int64_preserves_dtype(self):
        # Both operands have int64 indices; the result must too.
        # (If the fallback accidentally truncated to int32, int(indices[1])
        # would silently wrap and give the wrong column.)
        a = self._make_int64_csr(0)
        b = self._make_int64_csr(_LARGE)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_values_correct(self):
        # Values at column positions 0 and _LARGE are preserved after addition.
        # After sort_indices(), col 0 is always at position 0 and col _LARGE
        # at position 1, so direct array access is safe.
        a = self._make_int64_csr(0, value=3.0)
        b = self._make_int64_csr(_LARGE, value=7.0)
        c = (a + b)
        c.sort_indices()
        assert c.nnz == 2
        assert int(c.indices[0]) == 0
        assert int(c.indices[1]) == _LARGE
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(7.0)

    def test_add_int64_overlapping_entries_summed(self):
        # When A and B share a (row, col) position, the fallback concatenates
        # both entries into a COO and relies on sum_duplicates() to merge them.
        a = self._make_int64_csr(_LARGE, value=2.0)
        b = self._make_int64_csr(_LARGE, value=5.0)
        c = a + b
        assert c.nnz == 1
        assert c.indices.dtype == cupy.int64
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(7.0)

    def test_add_int64_alpha_beta_scaling(self):
        # _cupy_csrgeam_int64 scales a.data by alpha and b.data by beta before
        # concatenation.  Verify through the direct cusparse.csrgeam2 interface
        # since the __add__ operator always uses alpha=1, beta=1.
        # (This test caught a bug where _numpy.array(alpha, ...) returned a
        # 0-d ndarray that CuPy's __mul__ rejected with TypeError.)
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=1.0)
        c = cusparse.csrgeam2(a, b, alpha=3.0, beta=4.0)
        c.sort_indices()
        assert c.nnz == 2
        # col 0 → alpha*1.0 = 3.0;  col _LARGE → beta*1.0 = 4.0.
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(4.0)

    def test_spgeam_int64_fallback(self):
        # cusparse.spgeam() routes int64 directly to _cupy_csrgeam_int64 when
        # cusparseSpGEAM is unavailable (absent from all public releases ≤12.7.9).
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = cusparse.spgeam(a, b)
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_multirow(self):
        # The searchsorted(indptr[1:], arange(nnz)) expansion must assign
        # each nonzero to the correct row.  a has entries in both rows;
        # b has an entry only in row 0 that overlaps a's row-0 entry.
        a = sparse.csr_matrix(
            (cupy.array([1.0, 2.0]),
             cupy.array([0, _LARGE], dtype=cupy.int64),
             cupy.array([0, 1, 2], dtype=cupy.int64)),   # 1 nnz per row
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=3.0)   # row 0 → col _LARGE
        c = a + b
        c.sort_indices()
        assert c.nnz == 3
        # Row 0: cols 0 and _LARGE (2 entries).  Row 1: col _LARGE (1 entry).
        assert int(c.indptr[1]) == 2
        assert int(c.indptr[2]) == 3
        # Row 1's entry must retain the exact int64 column value.
        assert int(c.indices[2]) == _LARGE
        assert float(c.data[2]) == pytest.approx(2.0)

    def test_add_mixed_dtype_int32_plus_int64_promotes(self):
        # idx_dtype = numpy.result_type(int32, int64) == int64.
        # The int32 matrix has small column values; the int64 matrix has _LARGE.
        # The result must use int64 to represent _LARGE.
        data = cupy.array([1.0])
        a = sparse.csr_matrix(
            (data, cupy.array([5], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_empty_operand(self):
        # When one matrix has nnz=0, _cupy_csrgeam_int64 enters the
        # `a_rows = cupy.empty(0, idx_dtype)` branch.  The result equals
        # the non-empty matrix.
        a = self._make_int64_csr(_LARGE)
        b = sparse.csr_matrix(
            (cupy.empty(0, cupy.float64),
             cupy.empty(0, cupy.int64),
             cupy.zeros(3, cupy.int64)),
            shape=self._shape)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE

    def test_add_int32_regression(self):
        # int32 + int32 must continue to use the cuSPARSE (csrgeam2) path and
        # return int32 indices with correct values.
        data = cupy.array([1.0, 2.0])
        a = sparse.csr_matrix(
            (data[:1], cupy.array([0], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=(2, 4))
        b = sparse.csr_matrix(
            (data[1:], cupy.array([3], dtype=cupy.int32),
             cupy.array([0, 0, 1], dtype=cupy.int32)),
            shape=(2, 4))
        c = a + b
        assert c.indices.dtype == cupy.int32
        assert c.nnz == 2
        testing.assert_array_equal(c.toarray(), a.toarray() + b.toarray())


class TestInt64ScalarIndex:
    """Scalar index m[i, j] with int64 column/row > INT32_MAX.

    _compress_getitem_kern previously typed 'minor' as int32, silently
    truncating large column values so the equality check always failed and
    the lookup returned 0.  The fix changes int32 minor to S minor, matching
    the dtype of the ind (column/row index) array.
    """

    def _make_int64_csr(self, col=_LARGE, value=5.0):
        """Single nonzero at (row=0, col=col)."""
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def test_csr_scalar_index_large_col_returns_value(self):
        # m[0, _LARGE] must return the stored value, not 0.
        # Previously, int32 minor truncated _LARGE to a negative int32,
        # ind == minor was always False, and m[0, _LARGE] silently returned 0.
        m = self._make_int64_csr()
        result = m[0, _LARGE]
        assert float(result) == pytest.approx(5.0)

    def test_csr_scalar_index_absent_large_col_returns_zero(self):
        # Absence (structural zero) at a large column must return 0.
        m = self._make_int64_csr(col=_LARGE - 1)  # stored at _LARGE-1, not _LARGE
        result = m[0, _LARGE]
        assert float(result) == pytest.approx(0.0)

    def test_csr_scalar_index_small_col_int32_regression(self):
        # int32 matrix: scalar index must remain correct.
        data = cupy.array([7.0])
        indices = cupy.array([3], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        assert float(m[0, 3]) == pytest.approx(7.0)
        assert float(m[0, 4]) == pytest.approx(0.0)

    def test_csc_scalar_index_large_row_returns_value(self):
        # CSC m[row, col] uses the same kernel with minor = target row.
        # Verify the fix works for CSC too.
        data = cupy.array([3.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)  # row index
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)  # 1 nnz in col 0
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 1, 2))
        assert float(m[_LARGE, 0]) == pytest.approx(3.0)
        assert float(m[_LARGE - 1, 0]) == pytest.approx(0.0)

    def test_csr_complex_scalar_index_large_col(self):
        # Complex variant uses _compress_getitem_complex_kern, same int32 fix.
        data = cupy.array([2.0 + 3.0j])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        result = complex(m[0, _LARGE])
        assert result.real == pytest.approx(2.0)
        assert result.imag == pytest.approx(3.0)

    def test_csr_scalar_index_multiple_rows(self):
        # Matrix with one nonzero per row; index into both rows.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        assert float(m[0, 0]) == pytest.approx(1.0)
        assert float(m[1, _LARGE]) == pytest.approx(2.0)
        assert float(m[0, _LARGE]) == pytest.approx(0.0)
        assert float(m[1, 0]) == pytest.approx(0.0)


class TestInt64FancyRowIndex:
    """Fancy row index m[[r1, r2], :] with int64 column indices > INT32_MAX.

    Two int64 fixes work together here:
    - _csr_row_index_ker: all int32 parameters changed to I so int64 column
      values are not truncated in the output matrix.
    - _csr_indptr_to_coo_rows: xcsr2coo is int32-only; int64 path uses
      searchsorted to expand indptr to per-nnz row assignments.
    """

    _shape = (4, _LARGE + 2)

    def _make_int64_csr(self):
        """4-row CSR: row 0 → col 0 (val=1), row 2 → col _LARGE (val=2),
        rows 1 and 3 are empty."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1, 2, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def test_fancy_row_result_has_int64_indices(self):
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert sub.indices.dtype == cupy.int64
        assert sub.indptr.dtype == cupy.int64

    def test_fancy_row_large_col_preserved(self):
        # Previously, _csr_row_index_ker wrote Bj as int32, truncating
        # _LARGE to its low 32 bits (wrong negative int).
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert sub.nnz == 2
        assert int(sub.indices[0]) == 0
        assert int(sub.indices[1]) == _LARGE

    def test_fancy_row_values_correct(self):
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert float(sub.data[0]) == pytest.approx(1.0)
        assert float(sub.data[1]) == pytest.approx(2.0)

    def test_fancy_row_reverse_order(self):
        # Rows requested in reverse order: result should have reversed rows.
        m = self._make_int64_csr()
        sub = m[[2, 0], :]
        assert sub.nnz == 2
        assert int(sub.indices[0]) == _LARGE
        assert int(sub.indices[1]) == 0
        assert float(sub.data[0]) == pytest.approx(2.0)
        assert float(sub.data[1]) == pytest.approx(1.0)

    def test_fancy_row_single_row(self):
        m = self._make_int64_csr()
        sub = m[[2], :]
        assert sub.nnz == 1
        assert int(sub.indices[0]) == _LARGE
        assert sub.indices.dtype == cupy.int64

    def test_fancy_row_empty_rows_selected(self):
        # Selecting only empty rows should produce an empty matrix with
        # correct int64 dtypes.
        m = self._make_int64_csr()
        sub = m[[1, 3], :]
        assert sub.nnz == 0
        assert sub.indices.dtype == cupy.int64
        assert sub.indptr.dtype == cupy.int64

    def test_fancy_row_indptr_correct(self):
        # indptr[r+1] - indptr[r] must equal the nnz for each selected row.
        m = self._make_int64_csr()
        sub = m[[0, 1, 2, 3], :]  # all rows
        assert int(sub.indptr[0]) == 0
        assert int(sub.indptr[1]) == 1   # row 0 has 1 nnz
        assert int(sub.indptr[2]) == 1   # row 1 has 0 nnz
        assert int(sub.indptr[3]) == 2   # row 2 has 1 nnz
        assert int(sub.indptr[4]) == 2   # row 3 has 0 nnz

    def test_fancy_row_int32_regression(self):
        # int32 path must remain correct.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 5))
        sub = m[[2, 0], :]
        assert sub.indices.dtype == cupy.int32
        assert int(sub.indices[0]) == 1  # row 2 has col 1
        assert int(sub.indices[1]) == 0  # row 0 has col 0
        assert float(sub.data[0]) == pytest.approx(3.0)
        assert float(sub.data[1]) == pytest.approx(1.0)

    def test_fancy_row_dtype_not_demoted_when_values_small(self):
        # int64 dtype is preserved even when all values fit in int32.
        # The dtype is a property of the matrix, not the values.
        data = cupy.array([1.0])
        indices = cupy.array([5], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        sub = m[[0], :]
        assert sub.indices.dtype == cupy.int64


class TestInt64Argmax:
    """argmax/argmin along an axis with int64 column indices > INT32_MAX.

    _argmax_argmin_code previously used int* for the indices and indptr
    slice arrays, silently truncating int64 column values.  We added a
    TI template parameter so all index-typed values use the correct dtype.
    """

    def _make_int64_csr(self, nrows=3):
        """CSR matrix with one nonzero per row:
          row 0 → (col=0, val=1.0)
          row 1 → (col=_LARGE, val=2.0)   ← argmax column for float comparison
          row 2 → (col=_LARGE//2, val=1.5)
        """
        data = cupy.array([1.0, 2.0, 1.5])[:nrows]
        cols = [0, _LARGE, _LARGE // 2]
        indices = cupy.array(cols[:nrows], dtype=cupy.int64)
        indptr = cupy.arange(nrows + 1, dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(nrows, _LARGE + 1))

    def test_csr_argmax_axis1_large_col(self):
        # argmax(axis=1) must return the correct int64 column index.
        # Previously, int* indices truncated _LARGE, returning wrong col.
        m = self._make_int64_csr()
        result = m.argmax(axis=1)
        # Row 0: max at col 0.  Row 1: max at col _LARGE.  Row 2: max at col _LARGE//2.
        assert int(result[1, 0]) == _LARGE
        assert int(result[0, 0]) == 0

    def test_csr_argmin_axis1_large_col(self):
        # argmin(axis=1) where the minimum is a negative value at a large column.
        # Row has two nonzeros: col 1 → 1.0, col _LARGE → -3.0.
        # Implicit zeros at other cols are 0.0; -3.0 is the global min.
        data = cupy.array([1.0, -3.0])
        indices = cupy.array([1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(1, _LARGE + 1))
        result = m.argmin(axis=1)
        assert int(result[0, 0]) == _LARGE

    def test_csr_argmax_axis1_result_dtype(self):
        # Result dtype is int (default out dtype), not affected by index dtype.
        m = self._make_int64_csr()
        result = m.argmax(axis=1)
        assert result.dtype in (cupy.int32, cupy.int64, cupy.intp)

    def test_csc_argmax_axis0_large_row(self):
        # CSC argmax(axis=0) finds the row of the max per column.
        # With a large row index, TI=int64 must be used.
        data = cupy.array([3.0, 1.0])
        indices = cupy.array([_LARGE, 0], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per col
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 1, 2))
        result = m.argmax(axis=0)
        # Col 0: max is at row _LARGE.  Col 1: max is at row 0.
        assert int(result[0, 0]) == _LARGE
        assert int(result[0, 1]) == 0

    def test_csr_argmax_no_axis_flat_index(self):
        # argmax() with no axis returns a flat index.  This path goes through
        # COO conversion (int64-aware), not _arg_minor_reduce — so it worked
        # before too.  Include as a regression guard.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        ncols = _LARGE + 1
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, ncols))
        flat = int(m.argmax())
        r = flat // ncols
        c = flat % ncols
        assert r == 1
        assert c == _LARGE

    def test_csr_argmax_axis1_int32_regression(self):
        # int32 matrix: argmax must remain correct.
        data = cupy.array([1.0, 5.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 5))
        result = m.argmax(axis=1)
        assert int(result[0, 0]) == 0   # row 0: only col 0
        assert int(result[1, 0]) == 2   # row 1: only col 2
        assert int(result[2, 0]) == 1   # row 2: only col 1

    def test_csr_argmax_axis1_multiple_large_cols(self):
        # Multiple rows, each with the argmax at a large col.
        # Both rows have a nonzero at a large column.
        data = cupy.array([1.0, 2.0, 0.5, 3.0])
        indices = cupy.array([0, _LARGE, 0, _LARGE + 1], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 2))
        result = m.argmax(axis=1)
        assert int(result[0, 0]) == _LARGE     # max(1.0, 2.0) → col _LARGE
        assert int(result[1, 0]) == _LARGE + 1  # max(0.5, 3.0) → col _LARGE+1


class TestInt64EliminateZeros:
    """eliminate_zeros with int64 indices uses a pure-CuPy fallback.

    csr2csr_compress is int32-only (Legacy API). For int64 matrices we use
    boolean masking + searchsorted + unique/scatter.
    """

    def test_eliminate_zeros_removes_explicit_zeros(self):
        # Row 0: [1.0@0, 0.0@_LARGE, 2.0@_LARGE+1]; after: [1.0@0, 2.0@_LARGE+1]
        data = cupy.array([1.0, 0.0, 2.0])
        indices = cupy.array([0, _LARGE, _LARGE + 1], dtype=cupy.int64)
        indptr = cupy.array([0, 3, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 2))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int64
        testing.assert_array_equal(m.indices, cupy.array([0, _LARGE + 1],
                                                         dtype=cupy.int64))
        testing.assert_array_equal(m.data, cupy.array([1.0, 2.0]))
        testing.assert_array_equal(m.indptr,
                                   cupy.array([0, 2, 2], dtype=cupy.int64))

    def test_eliminate_zeros_all_nonzero_is_noop(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int64

    def test_eliminate_zeros_all_zero(self):
        data = cupy.array([0.0, 0.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        m.eliminate_zeros()
        assert m.nnz == 0
        assert m.indices.dtype == cupy.int64
        testing.assert_array_equal(m.indptr,
                                   cupy.array([0, 0], dtype=cupy.int64))

    def test_eliminate_zeros_int32_regression(self):
        # int32 path (csr2csr_compress) must still work.
        data = cupy.array([1.0, 0.0, 2.0])
        indices = cupy.array([0, 3, 5], dtype=cupy.int32)
        indptr = cupy.array([0, 3, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int32


class TestInt64Multiply:
    """Element-wise multiply for int64 matrices.

    cupy_multiply_by_dense and cupy_multiply_by_csr_step1/step2 previously
    had int32 shape parameters; they now use I (long long for int64 matrices).
    """

    def test_multiply_dense_broadcast_int64(self):
        # (1, _LARGE+1) sparse * (1, 1) dense — broadcasting.
        data = cupy.array([2.0, 3.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        sp = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        dn = cupy.full((1, 1), 4.0)
        result = sp.multiply(dn)
        assert result.nnz == 2
        assert result.indices.dtype == cupy.int64
        assert abs(float(result[0, 0]) - 8.0) < 1e-9
        assert abs(float(result[0, _LARGE]) - 12.0) < 1e-9

    def test_multiply_csr_int64(self):
        # element-wise sparse * sparse, same sparsity pattern.
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        a = sparse.csr_matrix(
            (cupy.array([2.0, 3.0]), indices, indptr),
            shape=(1, _LARGE + 1))
        b = sparse.csr_matrix(
            (cupy.array([4.0, 5.0]), indices.copy(), indptr.copy()),
            shape=(1, _LARGE + 1))
        result = a.multiply(b)
        assert result.nnz == 2
        assert result.indices.dtype == cupy.int64
        assert abs(float(result[0, 0]) - 8.0) < 1e-9
        assert abs(float(result[0, _LARGE]) - 15.0) < 1e-9

    def test_multiply_dense_int32_regression(self):
        # int32 matrix must still work (int32 shape params, no overflow risk).
        sp = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        dn = cupy.full((3, 3), 2.0)
        result = sp.multiply(dn)
        testing.assert_array_almost_equal(result.toarray(),
                                          2.0 * cupy.eye(3))

    def test_multiply_csr_int32_regression(self):
        # int32 × int32 sparse element-wise.
        a = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        b = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        result = a.multiply(b)
        assert result.indices.dtype == cupy.int32
        testing.assert_array_almost_equal(result.toarray(), cupy.eye(3))


class TestInt64Diagonal:
    """diagonal() for int64 matrices.

    _cupy_csr_diagonal previously had int32 rows/cols; now uses I.
    For a matrix with shape (few_rows, large_cols), diagonal() is practical
    (output has few_rows elements, no OOM).
    """

    def test_diagonal_int64_large_cols(self):
        # (2, _LARGE+1) matrix — diagonal is 2 elements.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 1], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        d = m.diagonal()
        assert d.shape == (2,)
        testing.assert_array_almost_equal(d, cupy.array([1.0, 2.0]))

    def test_diagonal_int64_absent_returns_zero(self):
        # Diagonal element at (1,1) is absent → 0.0.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        d = m.diagonal()
        assert abs(float(d[0]) - 1.0) < 1e-9
        assert abs(float(d[1]) - 0.0) < 1e-9

    def test_diagonal_int32_regression(self):
        # int32 path must still work.
        m = sparse.csr_matrix(cupy.eye(4, dtype=cupy.float64))
        d = m.diagonal()
        testing.assert_array_almost_equal(d, cupy.ones(4))


class TestInt64MinMaxReduction:
    """max/min axis-reductions with int64 index dtype.

    _max_min_reduction_code previously used int* for indptr slices and a plain
    int32 'length' parameter (= shape[axis]).  For int64 matrices with ncols >
    INT32_MAX, passing shape[1] to int32 raised OverflowError at kernel launch.
    The fix: RawModule + name_expressions templated on TI, dispatched by
    get_typename(self.indptr.dtype); shape param passed as idx_dtype.type(N).

    Design note: axis=1 (reduce over columns) on a CSR matrix sends
    length = shape[1] directly to the kernel — the critical int64 path.
    axis=0 converts to CSC first, then sends length = shape[0].
    Both paths exercise the same TI template, just with different shapes.
    """

    def _make_int64_csr(self):
        # 2 × (_LARGE+1) CSR — shape forces int64 index dtype.
        # Row 0: col 0 → 1.0, col 2 → 3.0
        # Row 1: col 1 → 2.0, col 2 → -1.0
        data = cupy.array([1.0, 3.0, 2.0, -1.0])
        indices = cupy.array([0, 2, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def test_max_axis1_int64(self):
        # Previously raised OverflowError: shape[1] = _LARGE+1 overflowed int32
        # in the kernel launch argument.  Now uses idx_dtype.type(N) = int64.
        m = self._make_int64_csr()
        assert m.indices.dtype == cupy.int64
        result = m.max(axis=1).toarray()
        # Row 0: max(1.0, 3.0, implicit zeros) = 3.0
        # Row 1: max(2.0, -1.0, implicit zeros) = 2.0
        assert float(result[0, 0]) == pytest.approx(3.0)
        assert float(result[1, 0]) == pytest.approx(2.0)

    def test_min_axis1_int64(self):
        m = self._make_int64_csr()
        assert m.indices.dtype == cupy.int64
        result = m.min(axis=1).toarray()
        # Row 0: min(1.0, 3.0, implicit zeros) = 0.0 (implicit zeros dominate)
        # Row 1: min(2.0, -1.0, implicit zeros) = -1.0
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[1, 0]) == pytest.approx(-1.0)

    def test_max_axis0_int64(self):
        # axis=0 on CSC: _minor_reduce receives length = shape[0] = _LARGE+1.
        # Construct (_LARGE+1, 2) CSC directly — indptr has 3 elements (tiny).
        # Col 0: rows 0,1 → 1.0, -2.0   Col 1: row 0 → 3.0
        data = cupy.array([1.0, -2.0, 3.0])
        indices = cupy.array([0, 1, 0], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        assert m.indices.dtype == cupy.int64
        result = m.max(axis=0).toarray()
        # Col 0: max(1.0, -2.0, implicit zeros) = 1.0
        # Col 1: max(3.0, implicit zeros) = 3.0
        assert float(result[0, 0]) == pytest.approx(1.0)
        assert float(result[0, 1]) == pytest.approx(3.0)

    def test_max_axis1_int32_regression(self):
        # int32 path must still work after the RawKernel → RawModule change.
        data = cupy.array([1.0, 3.0, 2.0, -1.0])
        indices = cupy.array([0, 2, 1, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 5))
        result = m.max(axis=1).toarray()
        assert float(result[0, 0]) == pytest.approx(3.0)
        assert float(result[1, 0]) == pytest.approx(2.0)


class TestInt64Toarray:
    """toarray() for int64 matrices.

    _cupy_csr2dense previously had int32 M, N shape parameters; for matrices
    where shape[1] > INT32_MAX, these caused OverflowError at the Python layer
    before the kernel was launched (numpy.int32(N) overflows for N > INT32_MAX).
    Now they use I (idx_dtype.type(N)), matching the index dtype.

    Practical constraint: a (1, _LARGE+1) dense output requires ~17 GB, so
    these tests use a try/except OOM guard and skip on 16 GB hardware.
    The int32 regression test always runs.
    """

    @testing.slow
    def test_toarray_int64_no_overflow_error(self):
        # Before fix: OverflowError at numpy.int32(_LARGE+1) in kernel args.
        # After fix: either succeeds (≥17 GB GPU) or OOMs gracefully.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        try:
            arr = m.toarray()
            assert arr.shape == (1, _LARGE + 1)
            assert float(arr[0, 0]) == pytest.approx(1.0)
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory for dense output')

    @testing.slow
    def test_toarray_order_f_no_overflow_error(self):
        # order='F' calls _cupy_csr2dense with row_major=False.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        try:
            arr = m.toarray(order='F')
            assert arr.shape == (1, _LARGE + 1)
            assert float(arr[0, 0]) == pytest.approx(1.0)
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory for dense output')

    def test_toarray_int32_regression(self):
        # int32 path must continue to work after the M, N parameter change.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        arr = m.toarray()
        expected = cupy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        testing.assert_array_almost_equal(arr, expected)


class TestInt64SpGEMM:
    """int64-related fixes to spgemm.

    spgemm previously hardcoded 'i' (int32) for the c_indices allocation.
    We updated this to numpy.result_type(a.indices, b.indices).

    cuSPARSE spgemm does not support int64 inputs on current releases; calling
    it with int64 previously gave a cryptic CUSPARSE_STATUS_NOT_SUPPORTED.
    We added an explicit ValueError guard (checked before format/shape checks)
    so callers get a clear error message.
    """

    def test_spgemm_rejects_int64(self):
        # The int64 guard fires before the has_canonical_format assert and the
        # shape-compatibility check, so any int64 CSR matrix suffices here.
        if not cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available')
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1], dtype=cupy.int64)
        a = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        with pytest.raises(ValueError, match='int64'):
            cusparse.spgemm(a, a)

    def test_spgemm_int32_result_dtype_preserved(self):
        # result_type(int32, int32) == int32: int64 work must not
        # regress the int32 path.
        if not cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available')
        a = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        b = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        c = a @ b
        assert c.indices.dtype == cupy.int32
        testing.assert_array_almost_equal(c.toarray(), cupy.eye(3))
