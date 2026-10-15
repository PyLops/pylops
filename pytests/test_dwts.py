import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import DWT, DWT2D, DWTND
from pylops.utils import dottest

par1 = {
    "ny": 7,
    "nx": 9,
    "nz": 9,
    "nt": 10,
    "imag": 0,
    "dtype": "float32",
}  # real (fp32)
par2 = {
    "ny": 7,
    "nx": 9,
    "nz": 9,
    "nt": 10,
    "imag": 0,
    "dtype": "float64",
}  # real (fp64)
par3 = {
    "ny": 7,
    "nx": 9,
    "nz": 9,
    "nt": 10,
    "imag": 1j,
    "dtype": "complex64",
}  # complex (cp64)
par4 = {
    "ny": 7,
    "nx": 9,
    "nz": 9,
    "nt": 10,
    "imag": 1j,
    "dtype": "complex128",
}  # complex (cp128)

np.random.seed(10)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
def test_unknown_wavelet(par):
    """Check error is raised if unknown wavelet is chosen is passed"""
    with pytest.raises(ValueError, match="not in family set"):
        _ = DWT(dims=par["nt"], wavelet="foo")
    with pytest.raises(ValueError, match="not in family set"):
        _ = DWT2D(dims=par["nt"], wavelet="foo")
    with pytest.raises(ValueError, match="not in family set"):
        _ = DWTND(dims=par["nt"], wavelet="foo")


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
def test_wrong_level(par):
    """Check error is raised if level is smaller than 0"""
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = DWT(dims=par["nt"], level=-1)
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = DWT2D(dims=(par["nt"], par["nx"]), level=-1)
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = DWTND(dims=(par["nt"], par["nx"], par["ny"]), level=-1)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_DWT_1dsignal(par):
    """Dot-test and inversion for DWT operator for 1d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWT(dims=[par["nt"]], axis=0, wavelet="haar", level=3, dtype=par["dtype"])
    x = np.random.normal(0.0, 1.0, par["nt"]).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, par["nt"]).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x, xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x, xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("axis", [0, 1])
def test_DWT_2dsignal(par, axis):
    """Dot-test and inversion for DWT operator for 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWT(
        dims=(par["nt"], par["nx"]),
        axis=axis,
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"])).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_DWT_3dsignal(par, axis):
    """Dot-test and inversion for DWT operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    DWTop = DWT(
        dims=(par["nt"], par["nx"], par["ny"]),
        axis=axis,
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"])).astype(
        dtype
    ) + par["imag"] * np.random.normal(
        0.0, 1.0, (par["nt"], par["nx"], par["ny"])
    ).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_DWT2D_2dsignal(par):
    """Dot-test and inversion for DWT2D operator for 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWT2D(
        dims=(par["nt"], par["nx"]),
        axes=(0, 1),
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"])).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
def test_DWT2D_3dsignal(par, axes):
    """Dot-test and inversion for DWT operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWT2D(
        dims=(par["nt"], par["nx"], par["ny"]),
        axes=axes,
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"])).astype(
        dtype
    ) + par["imag"] * np.random.normal(
        0.0, 1.0, (par["nt"], par["nx"], par["ny"])
    ).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_DWTND_3dsignal(par):
    """Dot-test and inversion for DWTND operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWTND(
        dims=(par["nt"], par["nx"], par["ny"]),
        axes=(0, 1, 2),
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"])).astype(
        dtype
    ) + par["imag"] * np.random.normal(
        0.0, 1.0, (par["nt"], par["nx"], par["ny"])
    ).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize(
    "axes", [(0, 1, 2), (0, 2, 3), (1, 2, 3), (0, 1, 3), (0, 1, 2, 3)]
)
def test_DWTND_4dsignal(par, axes):
    """Dot-test and inversion for DWTND operator for 4d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    DWTop = DWTND(
        dims=(par["nt"], par["nx"], par["ny"], par["nz"]),
        axes=axes,
        wavelet="haar",
        level=3,
        dtype=par["dtype"],
    )
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"], par["nz"])).astype(
        dtype
    ) + par["imag"] * np.random.normal(
        0.0, 1.0, (par["nt"], par["nx"], par["ny"], par["nz"])
    ).astype(dtype)

    assert dottest(
        DWTop,
        DWTop.shape[0],
        DWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)
