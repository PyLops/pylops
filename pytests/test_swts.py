import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import SWT, SWT2D, SWTND
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
        _ = SWT(dims=par["nt"], wavelet="foo")
    with pytest.raises(ValueError, match="not in family set"):
        _ = SWT2D(dims=(par["nt"], par["nx"]), wavelet="foo")
    with pytest.raises(ValueError, match="not in family set"):
        _ = SWTND(dims=(par["nt"], par["nx"], par["ny"]), wavelet="foo")


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
def test_wrong_level(par):
    """Check error is raised if level is smaller than 1"""
    with pytest.raises(ValueError, match="must be >= 1"):
        _ = SWT(dims=par["nt"], level=0)
    with pytest.raises(ValueError, match="must be >= 1"):
        _ = SWT2D(dims=(par["nt"], par["nx"]), level=0)
    with pytest.raises(ValueError, match="must be >= 1"):
        _ = SWTND(dims=(par["nt"], par["nx"], par["ny"]), level=0)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWT_1dsignal(par):
    """Dot-test and inversion for SWT operator for 1d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    SWTop = SWT(dims=[par["nt"]], axis=0, wavelet="haar", level=3, dtype=par["dtype"])
    x = np.random.normal(0.0, 1.0, par["nt"]).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, par["nt"]).astype(dtype)

    assert dottest(
        SWTop,
        SWTop.shape[0],
        SWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = (SWTop * x).ravel()
    xadj = SWTop.H * y  # adjoint is same as inverse for swt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x, xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x, xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWT_2dsignal(par):
    """Dot-test and inversion for SWT operator for 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    for axis in [0, 1]:
        SWTop = SWT(
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
            SWTop,
            SWTop.shape[0],
            SWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        y = SWTop * x.ravel()
        xadj = SWTop.H * y  # adjoint is same as inverse for swt
        assert y.dtype == par["dtype"]
        assert xadj.dtype == par["dtype"]

        xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
        assert_array_almost_equal(
            x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8
        )
        assert_array_almost_equal(
            x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWT_3dsignal(par):
    """Dot-test and inversion for SWT operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    for axis in [0, 1, 2]:
        SWTop = SWT(
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
            SWTop,
            SWTop.shape[0],
            SWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        y = SWTop * x.ravel()
        xadj = SWTop.H * y  # adjoint is same as inverse for swt
        assert y.dtype == par["dtype"]
        assert xadj.dtype == par["dtype"]

        xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
        assert_array_almost_equal(
            x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8
        )
        assert_array_almost_equal(
            x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWT2D_2dsignal(par):
    """Dot-test and inversion for SWT2D operator for 2d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    SWTop = SWT2D(
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
        SWTop,
        SWTop.shape[0],
        SWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = SWTop * x.ravel()
    xadj = SWTop.H * y  # adjoint is same as inverse for swt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWT2D_3dsignal(par):
    """Dot-test and inversion for SWT operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    for axes in [(0, 1), (0, 2), (1, 2)]:
        SWTop = SWT2D(
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
            SWTop,
            SWTop.shape[0],
            SWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        y = SWTop * x.ravel()
        xadj = SWTop.H * y  # adjoint is same as inverse for swt
        assert y.dtype == par["dtype"]
        assert xadj.dtype == par["dtype"]

        xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
        assert_array_almost_equal(
            x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8
        )
        assert_array_almost_equal(
            x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWTND_3dsignal(par):
    """Dot-test and inversion for SWTND operator for 3d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    SWTop = SWTND(
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
        SWTop,
        SWTop.shape[0],
        SWTop.shape[1],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    y = SWTop * x.ravel()
    xadj = SWTop.H * y  # adjoint is same as inverse for swt
    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]

    xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SWTND_4dsignal(par):
    """Dot-test and inversion for SWTND operator for 4d signal"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    for axes in [(0, 1, 2), (0, 2, 3), (1, 2, 3), (0, 1, 3), (0, 1, 2, 3)]:
        SWTop = SWTND(
            dims=(par["nt"], par["nx"], par["ny"], par["nz"]),
            axes=axes,
            wavelet="haar",
            level=3,
            dtype=par["dtype"],
        )
        x = np.random.normal(
            0.0, 1.0, (par["nt"], par["nx"], par["ny"], par["nz"])
        ).astype(dtype) + par["imag"] * np.random.normal(
            0.0, 1.0, (par["nt"], par["nx"], par["ny"], par["nz"])
        ).astype(dtype)

        assert dottest(
            SWTop,
            SWTop.shape[0],
            SWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        y = SWTop * x.ravel()
        xadj = SWTop.H * y  # adjoint is same as inverse for swt
        assert y.dtype == par["dtype"]
        assert xadj.dtype == par["dtype"]

        xinv = lsqr(SWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]
        assert_array_almost_equal(
            x.ravel(), xadj, decimal=4 if dtype == np.float32 else 8
        )
        assert_array_almost_equal(
            x.ravel(), xinv, decimal=4 if dtype == np.float32 else 8
        )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("wavelet", ["db3", "bior2.2", "rbio3.3"])
def test_SWTs_wavelets(par, wavelet):
    """Dot-test for SWT, SWT2D and SWTND operators with non-haar wavelets"""
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype
    dims = (par["nt"], par["nx"], par["ny"])

    for SWTop in [
        SWT(dims=dims, axis=1, wavelet=wavelet, level=2, dtype=par["dtype"]),
        SWT2D(dims=dims, axes=(0, 2), wavelet=wavelet, level=2, dtype=par["dtype"]),
        SWTND(dims=dims, axes=(0, 1, 2), wavelet=wavelet, level=2, dtype=par["dtype"]),
    ]:
        assert dottest(
            SWTop,
            SWTop.shape[0],
            SWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )
