import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr as sp_lsqr

from pylops.basicoperators import MatrixMult
from pylops.optimization.solver import cg, cgls, lsqr

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
    xinv = cg(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


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
    xinv = cgls(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
def test_lsqr(par):
    """Compare local Pylops and scipy LSQR"""
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
    xinv = lsqr(Aop, y, x0, niter=par["nx"])[0]
    xinv_sp = sp_lsqr(Aop, y_sp, iter_lim=par["nx"])[0]
    if par["x0"]:
        xinv_sp += x0

    assert_array_almost_equal(xinv, x, decimal=4)
    assert_array_almost_equal(xinv_sp, x, decimal=4)
