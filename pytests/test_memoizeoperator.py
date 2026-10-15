import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import pytest

from pylops import MemoizeOperator
from pylops.basicoperators import Diagonal, MatrixMult, VStack
from pylops.optimization.basic import cgls

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float32"}  # square real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex64"}  # square imag


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_memoize_evals(par):
    """Check nevals counter when same model/data vectors are inputted
    to the operator
    """
    np.random.seed(0)
    A = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Aop = MatrixMult(A, dtype=par["dtype"])
    Amemop = MemoizeOperator(Aop, max_neval=2)

    # 1st evaluation
    Amemop * np.ones(par["nx"])
    assert Amemop.neval == 1
    # repeat 1st evaluation multiple times
    for _ in range(2):
        Amemop * np.ones(par["nx"])
    assert Amemop.neval == 1
    # 2nd evaluation
    Amemop * np.full(par["nx"], 2)  # same
    assert Amemop.neval == 2
    # 3rd evaluation (np.ones goes out of store)
    Amemop * np.full(par["nx"], 3)  # same
    assert Amemop.neval == 3
    # 4th evaluation
    Amemop * np.ones(par["nx"])
    assert Amemop.neval == 4


@pytest.mark.parametrize(
    "par",
    [
        (par1j),
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


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3)])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("adjoint_first", [False, True])
def test_memoize_direction(shape, complex_values, adjoint_first):
    """Forward/adjoint composition must not reuse an inverse result."""
    matrix = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float64).reshape(shape)
    if complex_values:
        matrix = matrix + 1j * matrix[::-1]
    op = MatrixMult(matrix, dtype=matrix.dtype)
    memo = MemoizeOperator(op, max_neval=2)
    first = op.H if adjoint_first else op
    memo_first = memo.H if adjoint_first else memo
    second = op if adjoint_first else op.H
    memo_second = memo if adjoint_first else memo.H
    vector = np.arange(1, first.shape[1] + 1, dtype=matrix.dtype)
    intermediate = memo_first @ vector
    expected = second @ (first @ vector)

    result = memo_second @ intermediate

    assert_array_almost_equal(result, expected, decimal=12)
    assert memo.neval == 2
    assert_array_almost_equal(memo_second @ intermediate, expected, decimal=12)
    assert memo.neval == 2
    assert_array_almost_equal(memo_first @ vector, first @ vector, decimal=12)
    assert memo.neval == 2


@pytest.mark.parametrize("max_neval", [1, 2, np.inf])
@pytest.mark.parametrize("complex_values", [False, True])
def test_memoize_cgls(max_neval, complex_values):
    """Memoizing a non-unitary operator must preserve a least-squares solve."""
    diagonal = np.array([2.0, 3.0, 4.0])
    if complex_values:
        diagonal = diagonal + 1j * np.array([1.0, -0.5, 2.0])
    op = Diagonal(diagonal)
    truth = np.array([1.0, -2.0, 0.5], dtype=diagonal.dtype)
    data = op @ truth
    memo = MemoizeOperator(op, max_neval=max_neval)

    result = cgls(memo, data, np.zeros_like(truth), niter=3, tol=1e-14)[0]

    assert_array_almost_equal(result, truth, decimal=12)
    assert len(memo.store) <= max_neval
    assert memo.neval > 1


def test_memoize_direction_eviction():
    """The cache budget remains shared across forward and adjoint evaluations."""
    op = Diagonal(np.array([2.0, 3.0]))
    memo = MemoizeOperator(op, max_neval=2)
    x = np.array([1.0, 2.0])
    y = op @ x
    memo @ x
    memo.H @ y
    memo @ (2 * x)
    assert memo.neval == 3
    assert len(memo.store) == 2
    assert_array_almost_equal(memo.H @ y, op.H @ y)
    assert memo.neval == 3
    assert_array_almost_equal(memo @ x, op @ x)
    assert memo.neval == 4
    assert len(memo.store) == 2
