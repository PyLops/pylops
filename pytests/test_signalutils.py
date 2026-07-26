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

from pylops.utils.signalprocessing import convmtx, nonstationary_convmtx, slope_estimate

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


def _plane_wave_2d(x, y, f, c, theta):
    """2D Plane wave modelling"""
    # Define x-y grid
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Slowness vector
    p = (np.cos(np.deg2rad(theta)) / c, np.sin(np.deg2rad(theta)) / c)

    # Construct plane wave
    pw = np.exp(-1j * (2 * np.pi * f * (-(p[0] * Y + p[1] * X))))
    pw = np.real(pw)

    return pw


def _plane_wave_3d(y, x, z, f, c, theta, phi):
    """2D Plane wave modelling"""
    # Define y-x-z grid
    Y, X, Z = np.meshgrid(y, x, z, indexing="ij")

    # Slowness vector
    p = (
        np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)) / c,
        np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)) / c,
        np.cos(np.deg2rad(theta)) / c,
    )  # slowness vector

    # Construct plane wave
    pw = np.exp(-1j * (2 * np.pi * f * (-(p[0] * Y + p[1] * X + p[2] * Z))))
    pw = np.real(pw)

    return pw


@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
@pytest.mark.parametrize("sparse", [False, True])
def test_convmtx(par, sparse):
    """Compare convmtx with np.convolve (small filter)"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    H = convmtx(h, par["nt"], par["nh"] // 2, sparse=sparse)

    y = np.convolve(x, h, mode="same")
    y1 = (H @ x)[: par["nt"]]
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par1j), (par2), (par2j)])
@pytest.mark.parametrize("sparse", [False, True])
def test_convmtx1(par, sparse):
    """Compare convmtx with np.convolve (large filter)"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    X = convmtx(
        x,
        par["nh"],
        par["nh"] // 2 - 1 if par["nh"] % 2 == 0 else par["nh"] // 2,
        sparse=sparse,
    )

    y = np.convolve(x, h, mode="same")
    y1 = (X @ h)[: par["nt"]]
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par1j)])
@pytest.mark.parametrize("sparse", [False, True])
def test_nonstationary_convmtx(par, sparse):
    """Compare nonstationary_convmtx with convmtx for stationary filter"""
    x = np.random.normal(0, 1, par["nt"]) + par["imag"] * np.random.normal(
        0, 1, par["nt"]
    )

    h = np.hanning(par["nh"])
    H = convmtx(
        h,
        par["nt"],
        par["nh"] // 2 - 1 if par["nh"] % 2 == 0 else par["nh"] // 2,
        sparse=sparse,
    )

    H1 = nonstationary_convmtx(
        np.repeat(h[:, np.newaxis], par["nt"], axis=1).T,
        par["nt"],
        hc=par["nh"] // 2,
        pad=(par["nt"], par["nt"]),
    )

    y = (H @ x)[: par["nt"]]
    y1 = np.dot(H1, x)
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("angle", [-45, -20, 0, 20, 45])
def test_slope_estimation_analytical_2d(angle):
    """Slope estimation using the Structure tensor algorithm for
    2D plane wave - test against analytical solution."""

    # Define x and y axes
    ox, dx, nx = 0, 5, 101
    oy, dy, ny = 0, 5, 101
    x, y = np.arange(nx) * dx + ox, np.arange(ny) * dy + oy

    # Compute plane wave
    f = 10  # frequency
    c = 1500  # Velocity
    pw = _plane_wave_2d(x, y, f, c, angle)

    # Slopes
    slopes, _ = slope_estimate(
        pw,
        smooth=11,
        eps=0.0,
        dips=False,
        anisotropies=False,
    )

    assert_array_almost_equal(np.median(slopes), np.tan(np.deg2rad(angle)), decimal=2)


@pytest.mark.parametrize("angle", [-45, -20, 0, 20, 45])
def test_slope_estimation_analytical_3d(angle):
    """Slope estimation using the Structure tensor algorithm for
    3D plane wave - test against analytical solution."""

    # Define x and y axes
    oy, dy, ny = 0, 5, 21
    ox, dx, nx = 0, 5, 51
    oz, dz, nz = 0, 5, 51
    y, x, z = np.arange(ny) * dy + oy, np.arange(nx) * dx + ox, np.arange(nz) * dz + oz

    # Compute plane wave
    f = 10  # frequency
    c = 1500  # Velocity
    pw = _plane_wave_3d(y, x, z, f, c, angle, phi=0.0)

    # Slopes
    slopes, _ = slope_estimate(
        pw,
        dy=1.0,
        smooth=11,
        eps=0.0,
        dips=False,
        anisotropies=False,
    )

    assert_array_almost_equal(
        np.median(slopes[1]), np.tan(np.deg2rad(angle)), decimal=2
    )


def test_slope_estimation_reg():
    """Slope estimation using the Structure tensor algorithm should
    apply regularisation (some slopes are set to zero)
    while dips should not use regularisation."""

    img_test = np.identity(20)  # generate test with -45° angle
    eps = 0.09  # set a regularisation parameter that will be exceeded

    slopes, _ = slope_estimate(img_test, dips=False, eps=eps)
    slopes_dips, _ = slope_estimate(img_test, dips=True, eps=eps)

    assert np.any(np.isclose(slopes, 0.0))
    assert not np.any(np.isclose(slopes_dips, 0.0))
