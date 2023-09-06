import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils.signalprocessing import convmtx, nonstationary_convmtx

par1 = {"nt": 51, "nh": 7, "imag": 0, "dtype": "float32"}  # odd sign, odd filt, real
par1j = {
    "nt": 51,
    "nh": 7,
    "imag": 1j,
    "dtype": "complex64",
}  # odd sign, odd filt, complex
par2 = {"nt": 50, "nh": 7, "imag": 0, "dtype": "float32"}  # even sign, odd filt, real
par2j = {
    "nt": 50,
    "nh": 7,
    "imag": 1j,
    "dtype": "complex64",
}  # even sign, odd filt, complex
par3 = {"nt": 51, "nh": 6, "imag": 0, "dtype": "float32"}  # odd sign, even filt, real
par3j = {
    "nt": 51,
    "nh": 6,
    "imag": 1j,
    "dtype": "complex64",
}  # odd sign, even filt, complex
par4 = {"nt": 50, "nh": 6, "imag": 0, "dtype": "float32"}  # even sign, even filt, real
par4j = {
    "nt": 50,
    "nh": 6,
    "imag": 1j,
    "dtype": "complex64",
}  # even sign, even filt, complex


np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_convmtx(par):
    """Compare convmtx with np.convolve (small filter)"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    H = convmtx(h, par["nt"], par["nh"] // 2)

    y = np.convolve(x, h, mode="same")
    y1 = np.dot(H[: par["nt"]], x)
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
def test_convmtx1(par):
    """Compare convmtx with np.convolve (large filter)"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    X = convmtx(
        x, par["nh"], par["nh"] // 2 - 1 if par["nh"] % 2 == 0 else par["nh"] // 2
    )

    y = np.convolve(x, h, mode="same")
    y1 = np.dot(X[: par["nt"]], h)
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_nonstationary_convmtx(par):
    """Compare nonstationary_convmtx with convmtx for stationary filter"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    H = convmtx(
        h, par["nt"], par["nh"] // 2 - 1 if par["nh"] % 2 == 0 else par["nh"] // 2
    )

    H1 = nonstationary_convmtx(
        np.repeat(h[:, np.newaxis], par["nt"], axis=1).T,
        par["nt"],
        hc=par["nh"] // 2,
        pad=(par["nt"], par["nt"]),
    )
    y = np.dot(H[: par["nt"]], x)
    y1 = np.dot(H1, x)
    assert_array_almost_equal(y, y1, decimal=4)
