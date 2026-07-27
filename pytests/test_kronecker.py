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

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}  # square real (fp64)
par2 = {"ny": 21, "nx": 11, "imag": 0, "dtype": "float64"}  # overdetermined real (fp64)
par1s = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float32"}  # square real (fp32)
par2s = {
    "ny": 21,
    "nx": 11,
    "imag": 0,
    "dtype": "float32",
}  # overdetermined real (fp32)
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square imag
par2j = {"ny": 21, "nx": 11, "imag": 1j, "dtype": "complex128"}  # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_Kroneker(par):
    """Dot-test, inversion and comparison with np.kron for Kronecker operator"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    G1 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
    G2 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
    x = np.ones(par["nx"] ** 2, dtype=dtype) + par["imag"] * np.ones(
        par["nx"] ** 2, dtype=dtype
    )

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
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    y = Kop * x
    xadj = Kop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    if backend == "numpy":  # cupy is not accurate enough for square systems
        xlsqr = lsqr(
            Kop,
            Kop * x,
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=1000,
            atol=0,
            btol=0,
            conlim=np.inf,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=2)

    # Comparison with numpy
    assert_array_almost_equal(np.kron(G1, G2), Kop * np.eye(par["nx"] ** 2), decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s)])
def test_Kroneker_Derivative(par):
    """Use Kronecker operator to apply the Derivative operator over one axis
    and compare with FirstDerivative(... axis=axis)
    """
    Dop = FirstDerivative(par["ny"], sampling=1, edge=True, dtype=par["dtype"])
    D2op = FirstDerivative(
        (par["ny"], par["nx"]), axis=0, sampling=1, edge=True, dtype=par["dtype"]
    )

    Kop = Kronecker(Dop, Identity(par["nx"], dtype=par["dtype"]), dtype=par["dtype"])

    x = np.zeros((par["ny"], par["nx"]), dtype=par["dtype"])
    x[par["ny"] // 2, par["nx"] // 2] = 1

    yk = Kop * x.ravel()
    xadjk = Kop.H * yk
    assert yk.dtype == par["dtype"]
    assert xadjk.dtype == par["dtype"]

    y = D2op * x.ravel()
    assert_array_equal(y, yk)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_Kroneker_multiproc_multithread(par):
    """Single and multiprocess/multithreading consistency for Kroneker operator"""
    for parallel_kind in ["multiproc", "multithread"]:
        np.random.seed(10)
        nproc = 2

        dtype = np.empty(0, dtype=par["dtype"]).real.dtype

        G1 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
            "imag"
        ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
        G2 = np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype) + par[
            "imag"
        ] * np.random.normal(0, 10, (par["ny"], par["nx"])).astype(dtype)
        x = np.ones((par["nx"] ** 2, 4), dtype=dtype) + par["imag"] * np.ones(
            (par["nx"] ** 2, 4), dtype=dtype
        )
        y = np.ones((par["ny"] ** 2, 4), dtype=dtype) + par["imag"] * np.ones(
            (par["ny"] ** 2, 4), dtype=dtype
        )

        Kop = Kronecker(
            MatrixMult(G1, dtype=par["dtype"]),
            MatrixMult(G2, dtype=par["dtype"]),
            dtype=par["dtype"],
        )
        Kmultiop = Kronecker(
            MatrixMult(G1, dtype=par["dtype"]),
            MatrixMult(G2, dtype=par["dtype"]),
            nproc=nproc,
            parallel_kind=parallel_kind,
            dtype=par["dtype"],
        )
        assert dottest(
            Kmultiop,
            par["ny"] ** 2,
            par["nx"] ** 2,
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        # forward
        assert_array_almost_equal(Kop * x, Kmultiop * x, decimal=4)
        # adjoint
        assert_array_almost_equal(Kop.H * y, Kmultiop.H * y, decimal=4)

        # close pool
        Kmultiop.close()
