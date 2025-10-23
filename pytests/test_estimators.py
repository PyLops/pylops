import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest
from numpy.testing import assert_almost_equal

from pylops.basicoperators import MatrixMult
from pylops.utils.backend import to_numpy

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

    trace_true = npp.trace(to_numpy(A))
    assert type(trace_true) == npp.dtype(dtype)

    trace_expl = Aop.trace(backend=backend)
    assert to_numpy(trace_expl).dtype == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl, decimal=5)

    # Hutchinson
    trace_est = Aop.trace(
        neval=10 * n,
        batch_size=n + 1,
        method="hutchinson",
        sampler=sampler,
        backend=backend,
    )
    assert to_numpy(trace_est).dtype == np.dtype(dtype)
    decimal = 5 if sampler == "unitvector" else -1
    assert_almost_equal(trace_true, trace_est, decimal=decimal)


@pytest.mark.parametrize("par", pars_hutchpp)
def test_trace_hutchpp(par):
    """Test Hutch++ estimator."""
    np.random.seed(10)
    n, dtype, sampler = par["n"], par["dtype"], par["sampler"]

    A = np.random.randn(n, n).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)

    trace_true = npp.trace(to_numpy(A))
    assert type(trace_true) == npp.dtype(dtype)

    trace_expl = Aop.trace(backend=backend)
    assert to_numpy(trace_expl).dtype == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl, decimal=5)

    # Hutch++
    trace_est = Aop.trace(
        neval=10 * n,
        method="hutch++",
        sampler=sampler,
        backend=backend,
    )
    assert to_numpy(trace_est).dtype == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_est, decimal=5)


@pytest.mark.parametrize("par", pars_hutchpp)
def test_trace_nahutchpp(par):
    """Test NA-Hutch++ estimator."""
    np.random.seed(10)
    n, dtype, sampler = par["n"], par["dtype"], par["sampler"]

    A = np.random.randn(n, n).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)

    trace_true = npp.trace(to_numpy(A))
    assert type(trace_true) == npp.dtype(dtype)

    trace_expl = Aop.trace(backend=backend)
    assert to_numpy(trace_expl).dtype == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_expl, decimal=5)

    # NA-Hutch++
    trace_est = Aop.trace(
        neval=10 * n,
        method="na-hutch++",
        sampler=sampler,
        backend=backend,
    )
    assert to_numpy(trace_est).dtype == np.dtype(dtype)
    assert_almost_equal(trace_true, trace_est, decimal=-1)
