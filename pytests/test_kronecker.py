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

from pylops.basicoperators import FirstDerivative, Identity, Kronecker, MatrixMult
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}  # square real
par2 = {"ny": 21, "nx": 11, "imag": 0, "dtype": "float64"}  # overdetermined real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square imag
par2j = {"ny": 21, "nx": 11, "imag": 1j, "dtype": "complex128"}  # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Kroneker(par):
    """Dot-test, inversion and comparison with np.kron for Kronecker operator"""
    np.random.seed(10)
    G1 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(par["dtype"])
    G2 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(par["dtype"])
    x = np.ones(par["nx"] ** 2) + par["imag"] * np.ones(par["nx"] ** 2)

    Kop = Kronecker(
        MatrixMult(G1, dtype=par["dtype"]),
        MatrixMult(G2, dtype=par["dtype"]),
        dtype=par["dtype"],
    )
    assert dottest(
        Kop,
        par["ny"] ** 2,
        par["nx"] ** 2,
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    if backend == "numpy":  # cupy is not accurate enough for square systems
        xlsqr = lsqr(
            Kop,
            Kop * x,
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=300,
            atol=0,
            btol=0,
            conlim=np.inf,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=2)

    # Comparison with numpy
    assert_array_almost_equal(np.kron(G1, G2), Kop * np.eye(par["nx"] ** 2), decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Kroneker_Derivative(par):
    """Use Kronecker operator to apply the Derivative operator over one axis
    and compare with FirstDerivative(... axis=axis)
    """
    Dop = FirstDerivative(par["ny"], sampling=1, edge=True, dtype="float32")
    D2op = FirstDerivative(
        (par["ny"], par["nx"]), axis=0, sampling=1, edge=True, dtype="float32"
    )

    Kop = Kronecker(Dop, Identity(par["nx"], dtype=par["dtype"]), dtype=par["dtype"])

    x = np.zeros((par["ny"], par["nx"])) + par["imag"] * np.zeros(
        (par["ny"], par["nx"])
    )
    x[par["ny"] // 2, par["nx"] // 2] = 1

    y = D2op * x.ravel()
    yk = Kop * x.ravel()
    assert_array_equal(y, yk)
