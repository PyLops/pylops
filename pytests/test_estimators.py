import pytest

import numpy as np

from numpy.testing import assert_almost_equal
from pylops.basicoperators import MatrixMult

SAMPLERS = ["gaussian", "rayleigh", "rademacher", "unitvector"]
DTYPES = ["float32", "float64"]
pars_hutchinson = [
    {"n": 100, "dtype": dtype, "sampler": sampler}
    for dtype in DTYPES
    for sampler in SAMPLERS
]

pars_hutchpp = [
    {"n": 100, "dtype": dtype, "sampler": sampler}
    for dtype in DTYPES
    for sampler in SAMPLERS[:-1]
]


@pytest.mark.parametrize("par", pars_hutchinson)
def test_trace_hutchison(par):
    """Test Hutchinson estimator."""
    np.random.seed(10)
    n, dtype, sampler = par["n"], par["dtype"], par["sampler"]

    A = np.random.randn(n, n).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)

    trace_true = np.trace(A)
    assert type(trace_true) == np.dtype(dtype)

    trace_expl = Aop.trace()
    assert type(trace_expl) == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl)

    # Hutchinson
    trace_est = Aop.trace(
        neval=10 * n,
        batch_size=n + 1,
        method="hutchinson",
        sampler=sampler,
    )
    assert type(trace_est) == np.dtype(dtype)
    decimal = 7 if sampler == "unitvector" else -1
    assert_almost_equal(trace_true, trace_est, decimal=decimal)


@pytest.mark.parametrize("par", pars_hutchpp)
def test_trace_hutchpp(par):
    """Test Hutch++ estimator."""
    np.random.seed(10)
    n, dtype, sampler = par["n"], par["dtype"], par["sampler"]

    A = np.random.randn(n, n).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)

    trace_true = np.trace(A)
    assert type(trace_true) == np.dtype(dtype)

    trace_expl = Aop.trace()
    assert type(trace_expl) == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl)

    # Hutch++
    trace_est = Aop.trace(
        neval=10 * n,
        method="hutch++",
        sampler=sampler,
    )
    assert type(trace_est) == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_est, decimal=5)


@pytest.mark.parametrize("par", pars_hutchpp)
def test_trace_nahutchpp(par):
    """Test NA-Hutch++ estimator."""
    np.random.seed(10)
    n, dtype, sampler = par["n"], par["dtype"], par["sampler"]

    A = np.random.randn(n, n).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)

    trace_true = np.trace(A)
    assert type(trace_true) == np.dtype(dtype)

    trace_expl = Aop.trace()
    assert type(trace_expl) == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl)

    # NA-Hutch++
    trace_est = Aop.trace(
        neval=10 * n,
        method="na-hutch++",
        sampler=sampler,
    )
    assert type(trace_est) == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_est, decimal=-1)
