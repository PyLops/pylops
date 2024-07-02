import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse.linalg import LinearOperator as spLinearOperator

import pylops
from pylops import LinearOperator
from pylops.basicoperators import (
    Diagonal,
    FirstDerivative,
    HStack,
    MatrixMult,
    Real,
    Symmetrize,
    VStack,
    Zero,
)
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}  # square real
par2 = {"ny": 21, "nx": 11, "imag": 0, "dtype": "float64"}  # overdetermined real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square imag
par2j = {"ny": 21, "nx": 11, "imag": 1j, "dtype": "complex128"}  # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j)])
def test_overloads(par):
    """Apply various overloaded operators (.H, -, +, *) and ensure that the
    returned operator is still of pylops LinearOperator type
    """
    diag = np.arange(par["nx"]) + par["imag"] * np.arange(par["nx"])
    Dop = Diagonal(diag, dtype=par["dtype"])

    # .H
    assert isinstance(Dop.H, LinearOperator)
    # negate
    assert isinstance(-Dop, LinearOperator)
    # multiply by scalar
    assert isinstance(2 * Dop, LinearOperator)
    # +
    assert isinstance(Dop + Dop, LinearOperator)
    # -
    assert isinstance(Dop - 2 * Dop, LinearOperator)
    # *
    assert isinstance(Dop * Dop, LinearOperator)
    # **
    assert isinstance(Dop**2, LinearOperator)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_scaled(par):
    """Verify that _ScaledLinearOperator produces the correct type based
    on its inputs types
    """
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
        diag = np.arange(par["nx"], dtype=dtype) + par["imag"] * np.arange(
            par["nx"], dtype=dtype
        )
        Dop = Diagonal(diag, dtype=dtype)
        Sop = 3.0 * Dop
        S1op = -3.0 * Dop
        S2op = Dop * 3.0
        S3op = Dop * -3.0
        assert Sop.dtype == dtype
        assert S1op.dtype == dtype
        assert S2op.dtype == dtype
        assert S3op.dtype == dtype


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_scipyop(par):
    """Verify interaction between pylops and scipy Linear operators"""

    class spDiag(spLinearOperator):
        def __init__(self, x):
            self.x = x
            self.shape = (len(x), len(x))
            self.dtype = x.dtype

        def _matvec(self, x):
            return x * self.x

        def _rmatvec(self, x):
            return x * self.x

    Dop = Diagonal(np.ones(5))
    Dspop = spDiag(2 * np.ones(5))

    # sum pylops to scipy linear ops
    assert isinstance(Dop + Dspop, LinearOperator)

    # multiply pylops to scipy linear ops
    assert isinstance(Dop * Dspop, LinearOperator)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_dense(par):
    """Dense matrix representation of square matrix"""
    diag = np.arange(par["nx"]) + par["imag"] * np.arange(par["nx"])
    D = np.diag(diag)
    Dop = Diagonal(diag, dtype=par["dtype"])
    assert_array_equal(Dop.todense(), D)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_dense_skinny(par):
    """Dense matrix representation of skinny matrix"""
    diag = np.arange(par["nx"]) + par["imag"] * np.arange(par["nx"])
    D = np.diag(diag)
    Dop = Diagonal(diag, dtype=par["dtype"])
    Zop = Zero(par["nx"], 3, dtype=par["dtype"])
    Op = HStack([Dop, Zop])
    OOp = np.hstack((D, np.zeros((par["nx"], 3))))
    assert_array_equal(Op.todense(), OOp)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j)])
def test_sparse(par):
    """Sparse matrix representation"""
    diag = np.arange(par["nx"]) + par["imag"] * np.arange(par["nx"])
    D = np.diag(diag)
    Dop = Diagonal(diag, dtype=par["dtype"])
    S = Dop.tosparse()
    assert_array_equal(S.toarray(), D)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j)])
def test_eigs(par):
    """Eigenvalues and condition number estimate with ARPACK"""
    # explicit=True
    diag = np.arange(par["nx"], 0, -1) + par["imag"] * np.arange(par["nx"], 0, -1)
    Op = MatrixMult(
        np.vstack((np.diag(diag), np.zeros((par["ny"] - par["nx"], par["nx"]))))
    )
    eigs = Op.eigs()
    assert_array_almost_equal(diag[: eigs.size], eigs, decimal=3)

    cond = Op.cond()
    assert_array_almost_equal(np.real(cond), par["nx"], decimal=3)

    # explicit=False
    Op = Diagonal(diag, dtype=par["dtype"])
    if par["ny"] > par["nx"]:
        Op = VStack([Op, Zero(par["ny"] - par["nx"], par["nx"])])
    eigs = Op.eigs()
    assert_array_almost_equal(diag[: eigs.size], eigs, decimal=3)

    # uselobpcg cannot be used for square non-symmetric complex matrices
    if np.iscomplex(Op):
        eigs1 = Op.eigs(uselobpcg=True)
        assert_array_almost_equal(eigs, eigs1, decimal=3)

    cond = Op.cond()
    assert_array_almost_equal(np.real(cond), par["nx"], decimal=3)

    if np.iscomplex(Op):
        cond1 = Op.cond(uselobpcg=True, niter=100)
        assert_array_almost_equal(np.real(cond), np.real(cond1), decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_conj(par):
    """Complex conjugation operator"""
    M = 1j * np.ones((par["ny"], par["nx"]))
    Op = MatrixMult(M, dtype=np.complex128)
    Opconj = Op.conj()

    x = np.arange(par["nx"]) + par["imag"] * np.arange(par["nx"])
    y = Opconj * x

    # forward
    assert_array_almost_equal(Opconj * x, np.dot(M.conj(), x))

    # adjoint
    assert_array_almost_equal(Opconj.H * y, np.dot(M.T, y))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_apply_columns_explicit(par):
    """Apply columns to explicit and non-explicit operator"""
    M = np.ones((par["ny"], par["nx"]))
    Mop = MatrixMult(M, dtype=par["dtype"])
    M1op = MatrixMult(M, dtype=par["dtype"])
    M1op.explicit = False
    cols = np.sort(np.random.permutation(np.arange(par["nx"]))[: par["nx"] // 2])

    Mcols = M[:, cols]
    Mcolsop = Mop.apply_columns(cols)
    M1colsop = M1op.apply_columns(cols)

    x = np.arange(len(cols))
    y = np.arange(par["ny"])

    # forward
    assert_array_almost_equal(Mcols @ x, Mcolsop.matvec(x))
    assert_array_almost_equal(Mcols @ x, M1colsop.matvec(x))

    # adjoint
    assert_array_almost_equal(Mcols.T @ y, Mcolsop.rmatvec(y))
    assert_array_almost_equal(Mcols.T @ y, M1colsop.rmatvec(y))


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_realimag(par):
    """Real/imag extraction"""
    M = np.random.normal(0, 1, (par["ny"], par["nx"])) + 1j * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    Op = MatrixMult(M, dtype=np.complex128)

    Opr = Op.toreal()
    Opi = Op.toimag()

    # forward
    x = np.arange(par["nx"])
    y = Op * x
    yr = Opr * x
    yi = Opi * x

    assert_array_equal(np.real(y), yr)
    assert_array_equal(np.imag(y), yi)

    # adjoint
    y = np.arange(par["ny"]) + 1j * np.arange(par["ny"])
    x = Op.H * y
    xr = Opr.H * y
    xi = Opi.H * y

    assert_array_equal(np.real(x), xr)
    assert_array_equal(np.imag(x), -xi)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_rlinear(par):
    """R-linear"""
    np.random.seed(123)
    if np.dtype(par["dtype"]).kind == "c":
        M = (
            np.random.randn(par["ny"], par["nx"])
            + 1j * np.random.randn(par["ny"], par["nx"])
        ).astype(par["dtype"])
    else:
        M = np.random.randn(par["ny"], par["nx"]).astype(par["dtype"])

    OpM = MatrixMult(M, dtype=par["dtype"])
    OpR_left = Real(par["ny"], dtype=par["dtype"])
    OpR_right = Real(par["nx"], dtype=par["dtype"])

    Op_left = OpR_left * OpM
    Op_right = OpM * OpR_right

    assert Op_left.clinear is False
    assert Op_right.clinear is False

    # forward
    x = np.random.randn(par["nx"])
    y_left = Op_left * x
    y_right = Op_right * x

    assert_array_equal(np.real(M @ x), y_left)
    assert_array_equal(M @ np.real(x), y_right)

    # adjoint (dot product test)
    for complexflag in range(4):
        assert dottest(Op_left, par["ny"], par["nx"], complexflag=complexflag)
        assert dottest(Op_right, par["ny"], par["nx"], complexflag=complexflag)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_copy_dims_dimsd(par):
    """Apply various overloaded operators (.H, -, +, *) and ensure that the
    dims and dimsd properties are propagated
    """
    Dy = FirstDerivative((par["ny"], par["nx"]), axis=0)
    Dx = FirstDerivative((par["ny"], par["nx"]), axis=-1)
    dims = (par["ny"], par["nx"])
    dimsd = (par["ny"], par["nx"])
    dimsd_sym = (2 * par["ny"] - 1, par["nx"])
    S = Symmetrize(dims, axis=0)

    # .H
    assert S.H.dims == dimsd_sym
    assert S.H.dimsd == dims
    # .T
    assert S.T.dims == dimsd_sym
    assert S.T.dimsd == dims
    # negate
    assert (-Dx).dims == dims
    assert (-Dx).dimsd == dimsd
    # multiply by scalar
    assert (2 * Dx).dims == dims
    assert (2 * Dx).dimsd == dimsd
    assert (Dx * 2).dims == dims
    assert (Dx * 2).dimsd == dimsd
    # +
    assert (Dx + Dy).dims == dims
    assert (Dx + Dy).dimsd == dimsd
    # -
    assert (Dx - 2 * Dy).dims == dims
    assert (Dx - 2 * Dy).dimsd == dimsd
    # *
    assert (S @ Dx).dims == dims
    assert (S @ Dx).dimsd == dimsd_sym
    # **
    assert (Dx**3).dims == dims
    assert (Dx**3).dimsd == dimsd


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_non_flattened_arrays(par):
    """Apply operators on arrays which are not 1D."""
    dims = (par["ny"], par["nx"])
    D = FirstDerivative(dims, axis=-1)
    S = Symmetrize(dims, axis=0)

    x_nd = np.random.randn(*dims)
    x_1d = x_nd.ravel()
    k = 4
    X_nd = np.repeat(x_nd[..., np.newaxis], k, axis=-1)
    X_1d = np.repeat(x_1d[..., np.newaxis], k, axis=-1)
    Y_D = np.empty((*D.dimsd, k))
    Y_S = np.empty((*S.dimsd, k))

    for i in range(k):
        Y_D[..., i] = D @ x_nd
        Y_S[..., i] = S @ D @ x_nd

    assert_array_equal((D @ x_1d).reshape(D.dimsd), D @ x_nd)
    assert_array_equal((S @ D @ x_1d).reshape(S.dimsd), S @ D @ x_nd)
    assert_array_equal(Y_D, D @ X_nd)
    assert_array_equal(Y_D, (D @ X_1d).reshape((*D.dimsd, -1)))
    assert_array_equal(Y_S, S @ D @ X_nd)
    assert_array_equal(Y_S, (S @ D @ X_1d).reshape((*S.dimsd, -1)))

    with pylops.disabled_ndarray_multiplication():
        with pytest.raises(ValueError):
            D @ x_nd
        with pytest.raises(ValueError):
            D @ X_nd


@pytest.mark.parametrize("par", [(par1), (par2j)])
def test_counts(par):
    """Assess counters and the associated reset method"""
    A = np.ones((par["ny"], par["nx"])) + par["imag"] * np.ones((par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    _ = Aop.matvec(np.ones(par["nx"]))
    _ = Aop.matvec(np.ones(par["nx"]))
    _ = Aop.rmatvec(np.ones(par["ny"]))
    _ = Aop @ np.ones(par["nx"])

    _ = Aop.rmatmat(np.ones((par["ny"], 2)))
    _ = Aop.matmat(np.ones((par["nx"], 2)))
    _ = Aop.rmatmat(np.ones((par["ny"], 2)))

    assert Aop.matvec_count == 3
    assert Aop.rmatvec_count == 1
    assert Aop.matmat_count == 1
    assert Aop.rmatmat_count == 2

    # reset
    Aop.reset_count()
    assert Aop.matvec_count == 0
    assert Aop.rmatvec_count == 0
    assert Aop.matmat_count == 0
    assert Aop.rmatmat_count == 0
