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

from pylops.basicoperators import Diagonal
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {"ny": 21, "nx": 11, "nt": 20, "imag": 0, "dtype": "float32"}  # real (fp32)
par1s = {"ny": 21, "nx": 11, "nt": 20, "imag": 0, "dtype": "float64"}  # real (fp64)
par1j = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
}  # complex (cp128)

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
@pytest.mark.parametrize("idim", [0, 1])
def test_Diagonal_1dsignal(par, idim):
    """Dot-test and inversion for Diagonal operator for 1d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    ddim = (par["nx"], par["nt"])[idim]
    d = (
        np.arange(ddim, dtype=dtype)
        + 1.0
        + par["imag"] * (np.arange(ddim, dtype=dtype) + 1.0)
    )

    Dop = Diagonal(d, dtype=par["dtype"])
    assert dottest(
        Dop,
        ddim,
        ddim,
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(ddim, dtype=dtype) + par["imag"] * np.ones(ddim, dtype=dtype)
    y = Dop * x
    xadj = Dop.H * y
    assert y.dtype == Dop.dtype
    assert xadj.dtype == par["dtype"]

    xlsqr = lsqr(
        Dop,
        Dop * x,
        x0=np.zeros_like(x).ravel(),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
@pytest.mark.parametrize("idim", [0, 1])
def test_Diagonal_2dsignal(par, idim):
    """Dot-test and inversion for Diagonal operator for 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    ddim = (par["nx"], par["nt"])[idim]
    d = (
        np.arange(ddim, dtype=dtype)
        + 1.0
        + par["imag"] * (np.arange(ddim, dtype=dtype) + 1.0)
    )

    Dop = Diagonal(d, dims=(par["nx"], par["nt"]), axis=idim, dtype=par["dtype"])
    assert dottest(
        Dop,
        par["nx"] * par["nt"],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["nx"], par["nt"])) + par["imag"] * np.ones((par["nx"], par["nt"]))
    xlsqr = lsqr(
        Dop,
        Dop * x.ravel(),
        x0=np.zeros_like(x).ravel(),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert_array_almost_equal(x.ravel(), xlsqr.ravel(), decimal=4)


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
@pytest.mark.parametrize("idim", [0, 1, 2])
def test_Diagonal_3dsignal(par, idim):
    """Dot-test and inversion for Diagonal operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    ddim = (par["ny"], par["nx"], par["nt"])[idim]
    d = (
        np.arange(ddim, dtype=dtype)
        + 1.0
        + par["imag"] * (np.arange(ddim, dtype=dtype) + 1.0)
    )

    Dop = Diagonal(
        d, dims=(par["ny"], par["nx"], par["nt"]), axis=idim, dtype=par["dtype"]
    )
    assert dottest(
        Dop,
        par["ny"] * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["ny"], par["nx"], par["nt"]), dtype=dtype) + par["imag"] * np.ones(
        (par["ny"], par["nx"], par["nt"]), dtype=dtype
    )
    y = Dop * x
    xadj = Dop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xlsqr = lsqr(
        Dop,
        Dop * x.ravel(),
        x0=np.zeros_like(x).ravel(),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(
        x.ravel(), xlsqr.ravel(), decimal=2 if dtype == np.float32 else 4
    )


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
@pytest.mark.parametrize("idim", [0, 1])
def test_Diagonal_2dsignal_unflattened(par, idim):
    """Dot-test and inversion for Diagonal operator for unflattened 2d signal (v2 behaviour)"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    ddim = (par["nx"], par["nt"])[idim]
    d = (
        np.arange(ddim, dtype=dtype)
        + 1.0
        + par["imag"] * (np.arange(ddim, dtype=dtype) + 1.0)
    )

    Dop = Diagonal(d, dims=(par["nx"], par["nt"]), axis=idim, dtype=par["dtype"])
    assert dottest(
        Dop,
        par["nx"] * par["nt"],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["nx"], par["nt"]), dtype=dtype) + par["imag"] * np.ones(
        (par["nx"], par["nt"]), dtype=dtype
    )
    y = Dop * x
    xadj = Dop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xlsqr = lsqr(
        Dop,
        Dop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
@pytest.mark.parametrize("idim", [0, 1, 2])
def test_Diagonal_3dsignal_unflattened(par, idim):
    """Dot-test and inversion for Diagonal operator unflattened 3d signal (v2 behaviour)"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    ddim = (par["ny"], par["nx"], par["nt"])[idim]
    d = (
        np.arange(ddim, dtype=dtype)
        + 1.0
        + par["imag"] * (np.arange(ddim, dtype=dtype) + 1.0)
    )

    Dop = Diagonal(
        d, dims=(par["ny"], par["nx"], par["nt"]), axis=idim, dtype=par["dtype"]
    )
    assert dottest(
        Dop,
        par["ny"] * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["ny"], par["nx"], par["nt"]), dtype=dtype) + par["imag"] * np.ones(
        (par["ny"], par["nx"], par["nt"]), dtype=dtype
    )
    y = Dop * x
    xadj = Dop.H * y
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xlsqr = lsqr(
        Dop,
        Dop * x,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=300,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)
