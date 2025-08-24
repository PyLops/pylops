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
from scipy.sparse.linalg import lsqr as sp_lsqr

from pylops.basicoperators import MatrixMult
from pylops.optimization.basic import cg, cgls, lsqr

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real, zero initial guess
par2 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # square real, non-zero initial guess
par3 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real, zero initial guess
par4 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # overdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # square complex, zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # square complex, non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # overdetermined complex, zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # overdetermined complex, non-zero initial guess


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cg(par):
    """CG with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cg(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, preallocate=preallocate)[0]
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cg_ndarray(par):
    """CG with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cg(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.shape
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cg_forceflat(par):
    """CG with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cg(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.ravel().shape
        assert_array_almost_equal(x.ravel(), xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cg_stopping(par):
    """CG testing stopping criterion rtol"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        rtol = 1e-2
        _, _, cost = cg(
            Aop, y, x0=x0, niter=par["nx"], tol=0, rtol=rtol, preallocate=preallocate
        )
        assert cost[-2] / cost[0] >= rtol
        assert cost[-1] / cost[0] < rtol


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cgls(par):
    """CGLS with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cgls(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, preallocate=preallocate)[
            0
        ]
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cgls_ndarray(par):
    """CGLS with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cgls(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.shape
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cgls_forceflat(par):
    """CGLS with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = cgls(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.ravel().shape
        assert_array_almost_equal(x.ravel(), xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_cgls_stopping(par):
    """CGLS testing stopping criterion rtol"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        rtol = 1e-2
        cost = cgls(
            Aop, y, x0=x0, niter=par["nx"], tol=0, rtol=rtol, preallocate=preallocate
        )[-1]
        assert cost[-2] / cost[0] >= rtol
        assert cost[-1] / cost[0] < rtol


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_lsqr_pylops_scipy(par):
    """Compare Pylops and scipy LSQR"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None
    y = Aop * x
    if par["x0"]:
        y_sp = y - Aop * x0
    else:
        y_sp = y.copy()

    xinv_sp = sp_lsqr(Aop, y_sp, iter_lim=par["nx"])[0]
    if par["x0"]:
        xinv_sp += x0

    for preallocate in [False, True]:
        xinv = lsqr(Aop, y, x0, niter=par["nx"], preallocate=preallocate)[0]

        assert_array_almost_equal(xinv, x, decimal=4)
        assert_array_almost_equal(xinv_sp, x, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_lsqr(par):
    """LSQR with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = lsqr(Aop, y, x0=x0, niter=par["nx"], atol=1e-5, preallocate=preallocate)[
            0
        ]
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_lsqr_ndarray(par):
    """LSQR with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = lsqr(Aop, y, x0=x0, niter=2 * x.size, atol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.shape
        assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_lsqr_forceflat(par):
    """LSQR with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    for preallocate in [False, True]:
        xinv = lsqr(Aop, y, x0=x0, niter=2 * x.size, atol=0, preallocate=preallocate)[0]
        assert xinv.shape == x.ravel().shape
        assert_array_almost_equal(x.ravel(), xinv, decimal=4)
