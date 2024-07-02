import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import Diagonal, HStack, Identity, MatrixMult, Smoothing1D
from pylops.optimization.leastsquares import (
    normal_equations_inversion,
    preconditioned_inversion,
    regularized_inversion,
)

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real with zero initial guess
par2 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # square real with non-zero initial guess
par3 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real with zero initial guess
par4 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # overdetermined real with non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # square complex with zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # square complex with non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # overdetermined complex with zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # overdetermined complex with non-zero
# initial guess


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_NormalEquationsInversion(par):
    """Solve normal equations in least squares sense"""
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, dtype=par["dtype"])

    Reg = MatrixMult(np.eye(par["nx"]), dtype=par["dtype"])
    NReg = MatrixMult(np.eye(par["nx"]), dtype=par["dtype"])
    Weigth = Diagonal(np.ones(par["ny"]), dtype=par["dtype"])
    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    x0 = (
        np.random.normal(0, 10, par["nx"])
        + par["imag"] * np.random.normal(0, 10, par["nx"])
        if par["x0"]
        else None
    )
    y = Gop * x

    # normal equations with regularization
    xinv = normal_equations_inversion(
        Gop, y, [Reg], epsI=1e-5, epsRs=[1e-8], x0=x0, **dict(maxiter=200, atol=1e-10)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)
    # normal equations with weight
    xinv = normal_equations_inversion(
        Gop, y, None, Weight=Weigth, epsI=1e-5, x0=x0, **dict(maxiter=200, atol=1e-10)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)
    # normal equations with weight and small regularization
    xinv = normal_equations_inversion(
        Gop,
        y,
        [Reg],
        Weight=Weigth,
        epsI=1e-5,
        epsRs=[1e-8],
        x0=x0,
        **dict(maxiter=200, atol=1e-10)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)
    # normal equations with weight and small normal regularization
    xinv = normal_equations_inversion(
        Gop,
        y,
        [],
        NRegs=[NReg],
        Weight=Weigth,
        epsI=1e-5,
        epsNRs=[1e-8],
        x0=x0,
        **dict(maxiter=200, atol=1e-10)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_RegularizedInversion(par):
    """Solve regularized inversion in least squares sense"""
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, dtype=par["dtype"])
    Reg = MatrixMult(np.eye(par["nx"]), dtype=par["dtype"])
    Weigth = Diagonal(np.ones(par["ny"]), dtype=par["dtype"])
    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    x0 = (
        np.random.normal(0, 10, par["nx"])
        + par["imag"] * np.random.normal(0, 10, par["nx"])
        if par["x0"]
        else None
    )
    y = Gop * x

    # regularized inversion with regularization
    xinv = regularized_inversion(
        Gop, y, [Reg], epsRs=[1e-8], x0=x0, **dict(damp=0, iter_lim=200, show=0)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)
    # regularized inversion with weight
    xinv = regularized_inversion(
        Gop, y, None, Weight=Weigth, x0=x0, **dict(damp=0, iter_lim=200, show=0)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)
    # regularized inversion with regularization
    xinv = regularized_inversion(
        Gop,
        y,
        [Reg],
        Weight=Weigth,
        epsRs=[1e-8],
        x0=x0,
        **dict(damp=0, iter_lim=200, show=0)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_WeightedInversion(par):
    """Compare results for normal equations and regularized inversion
    when used to solve weighted least square inversion
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, dtype=par["dtype"])
    w = np.arange(par["ny"])
    w1 = np.sqrt(w)
    Weigth = Diagonal(w, dtype=par["dtype"])
    Weigth1 = Diagonal(w1, dtype=par["dtype"])
    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Gop * x

    xne = normal_equations_inversion(
        Gop, y, None, Weight=Weigth, **dict(maxiter=5, atol=1e-10)
    )[0]
    xreg = regularized_inversion(
        Gop, y, None, Weight=Weigth1, **dict(damp=0, iter_lim=5, show=0)
    )[0]
    assert_array_almost_equal(xne, xreg, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
def test_PreconditionedInversion(par):
    """Solve regularized inversion in least squares sense"""
    np.random.seed(10)
    G = np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32") + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype("float32")
    Gop = MatrixMult(G, dtype=par["dtype"])

    Pre = Smoothing1D(nsmooth=5, dims=[par["nx"]], dtype=par["dtype"])
    p = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    x = Pre * p
    x0 = (
        np.random.normal(0, 1, par["nx"])
        + par["imag"] * np.random.normal(0, 1, par["nx"])
        if par["x0"]
        else None
    )
    y = Gop * x
    xinv = preconditioned_inversion(
        Gop, y, Pre, x0=x0, **dict(damp=0, iter_lim=800, show=0)
    )[0]
    assert_array_almost_equal(x, xinv, decimal=2)


@pytest.mark.parametrize("par", [(par1)])
def test_skinnyregularization(par):
    """Solve inversion with a skinny regularization (rows are smaller than
    the number of elements in the model vector)
    """
    np.random.seed(10)
    d = np.arange(par["nx"] - 1).astype(par["dtype"]) + 1.0
    Dop = Diagonal(d, dtype=par["dtype"])
    Regop = HStack([Identity(par["nx"] // 2), Identity(par["nx"] // 2)])

    x = np.arange(par["nx"] - 1)
    y = Dop * x

    xinv = normal_equations_inversion(Dop, y, [Regop], epsRs=[1e-4])[0]
    assert_array_almost_equal(x, xinv, decimal=2)

    xinv = regularized_inversion(Dop, y, [Regop], epsRs=[1e-4])[0]
    assert_array_almost_equal(x, xinv, decimal=2)
