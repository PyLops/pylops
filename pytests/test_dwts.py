import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import DWT, DWT2D
from pylops.utils import dottest

par1 = {"ny": 7, "nx": 9, "nt": 10, "imag": 0, "dtype": "float32"}  # real
par2 = {"ny": 7, "nx": 9, "nt": 10, "imag": 1j, "dtype": "complex64"}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_wavelet(par):
    """Check error is raised if unknown wavelet is chosen is passed"""
    with pytest.raises(ValueError):
        _ = DWT(dims=par["nt"], wavelet="foo")


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_1dsignal(par):
    """Dot-test and inversion for DWT operator for 1d signal"""
    DWTop = DWT(dims=[par["nt"]], axis=0, wavelet="haar", level=3)
    x = np.random.normal(0.0, 1.0, par["nt"]) + par["imag"] * np.random.normal(
        0.0, 1.0, par["nt"]
    )

    assert dottest(
        DWTop, DWTop.shape[0], DWTop.shape[1], complexflag=0 if par["imag"] == 0 else 3
    )

    y = DWTop * x
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=8)
    assert_array_almost_equal(x, xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_2dsignal(par):
    """Dot-test and inversion for DWT operator for 2d signal"""
    for axis in [0, 1]:
        DWTop = DWT(dims=(par["nt"], par["nx"]), axis=axis, wavelet="haar", level=3)
        x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"])) + par[
            "imag"
        ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"]))

        assert dottest(
            DWTop,
            DWTop.shape[0],
            DWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
        )

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_3dsignal(par):
    """Dot-test and inversion for DWT operator for 3d signal"""
    for axis in [0, 1, 2]:
        DWTop = DWT(
            dims=(par["nt"], par["nx"], par["ny"]), axis=axis, wavelet="haar", level=3
        )
        x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"])) + par[
            "imag"
        ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"]))

        assert dottest(
            DWTop,
            DWTop.shape[0],
            DWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
        )

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT2D_2dsignal(par):
    """Dot-test and inversion for DWT2D operator for 2d signal"""
    DWTop = DWT2D(dims=(par["nt"], par["nx"]), axes=(0, 1), wavelet="haar", level=3)
    x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"]))

    assert dottest(
        DWTop, DWTop.shape[0], DWTop.shape[1], complexflag=0 if par["imag"] == 0 else 3
    )

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]

    assert_array_almost_equal(x.ravel(), xadj, decimal=8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT2D_3dsignal(par):
    """Dot-test and inversion for DWT operator for 3d signal"""
    for axes in [(0, 1), (0, 2), (1, 2)]:
        DWTop = DWT2D(
            dims=(par["nt"], par["nx"], par["ny"]), axes=axes, wavelet="haar", level=3
        )
        x = np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"])) + par[
            "imag"
        ] * np.random.normal(0.0, 1.0, (par["nt"], par["nx"], par["ny"]))

        assert dottest(
            DWTop,
            DWTop.shape[0],
            DWTop.shape[1],
            complexflag=0 if par["imag"] == 0 else 3,
        )

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, atol=1e-8, btol=1e-8, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)
