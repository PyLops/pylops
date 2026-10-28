import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal, assert_array_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal, assert_array_equal

    backend = "numpy"
import pytest

from pylops import MemoizeOperator
from pylops.basicoperators import Diagonal, MatrixMult, VStack

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float32"}  # square real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex64"}  # square imag
par2 = {"ny": 11, "nx": 7, "imag": 0, "dtype": "float32"}  # rectangular real
par2j = {"ny": 11, "nx": 7, "imag": 1j, "dtype": "complex64"}  # rectangular imag


@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_memoize_evals(par, adjoint):
    """Check nevals counter when same model/data vectors are inputted
    to the operator
    """
    np.random.seed(0)
    A = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Aop = MatrixMult(A, dtype=par["dtype"])
    Amemop = MemoizeOperator(Aop, max_neval=2)

    if adjoint:
        Amemop1 = Amemop.H
        n = par["ny"]
    else:
        Amemop1 = Amemop
        n = par["nx"]

    # 1st evaluation
    Amemop1 * np.ones(n)
    assert Amemop.neval == 1
    # repeat 1st evaluation multiple times
    for _ in range(2):
        Amemop1 * np.ones(n)
    assert Amemop.neval == 1
    # 2nd evaluation
    Amemop1 * np.full(n, 2)  # same
    assert Amemop.neval == 2
    # 3rd evaluation (np.ones goes out of store)
    Amemop1 * np.full(n, 3)  # same
    assert Amemop.neval == 3
    # 4th evaluation
    Amemop1 * np.ones(n)
    assert Amemop.neval == 4


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_memoize_nearby_inputs(par, adjoint, dtype):
    """Check that nearby vectors (even within allclose)
    are identified as distinct evaluations."""
    Op = Diagonal(np.arange(par["nx"], dtype=dtype), dtype=dtype)
    MOp = MemoizeOperator(Op)

    Op = Op.H if adjoint else Op
    M1Op = MOp.H if adjoint else MOp
    x = np.ones(par["nx"], dtype=dtype)
    x_nearby = x.copy()
    x_nearby[0] += 1e-6
    assert np.allclose(x, x_nearby)

    # 2 evals with original and nearby input
    M1Op @ x
    y = M1Op @ x_nearby

    assert MOp.neval == 2
    assert_array_equal(y, Op @ x_nearby)
    assert_array_equal(y, M1Op @ x_nearby)


@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_memoize_input_dtype(par, adjoint):
    """Check that equal vectors with different precision
    are identified as distinct evaluations."""
    low = np.complex64 if par["imag"] == 1j else np.float32
    high = np.complex128 if par["imag"] == 1j else np.float64
    A = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")

    Aop = MatrixMult(A, dtype=low)
    Amemop = MemoizeOperator(Aop)

    Aop = Aop.H if adjoint else Aop
    Amemop1 = Amemop.H if adjoint else Amemop

    x = np.ones(par["nx"], dtype=low)
    x_promoted = x.astype(high)

    # 2 evals with original and promoted input
    Amemop1 @ x
    result = Amemop1 @ x_promoted
    expected = Aop @ x_promoted

    assert result.dtype == expected.dtype
    assert_array_equal(result, expected)
    assert Amemop.neval == 2


@pytest.mark.parametrize(
    "par",
    [
        (par1j),
        (par2j),
    ],
)
def test_memoize_evals_2(par):
    """Inversion of problem with real model and complex data, using two
    equivalent approaches: 1. complex operator enforcing the output of adjoint
    to be real, 2. joint system of equations for real and complex parts
    """
    np.random.seed(0)
    rdtype = np.real(np.ones(1, dtype=par["dtype"])).dtype
    A = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(rdtype) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(rdtype)
    Aop = MatrixMult(A, dtype=par["dtype"])
    x = np.ones(par["nx"], dtype=rdtype)
    y = Aop * x

    # Approach 1
    Aop1 = Aop.toreal(forw=False, adj=True)
    xinv1 = Aop1.div(y).real
    assert_array_almost_equal(xinv1, x, decimal=2 if rdtype == np.float32 else 4)

    # Approach 2
    Amop = MemoizeOperator(Aop, max_neval=10)
    Aop2 = VStack([Amop.toreal(), Amop.toimag()])
    y2 = np.concatenate([np.real(y), np.imag(y)])
    xinv2 = Aop2.div(y2)
    assert_array_almost_equal(xinv2, x, decimal=2 if rdtype == np.float32 else 4)
