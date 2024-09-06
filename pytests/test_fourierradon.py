import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FourierRadon2D, FourierRadon3D
from pylops.utils import dottest

par1 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "centeredh": True,
    "kind": "linear",
    "interp": True,
    "engine": "numpy",
}  # linear, numpy
par2 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "kind": "linear",
    "engine": "numba",
}  # linear, numba
par3 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 8e-3,
    "pxmax": 7e-3,
    "kind": "parabolic",
    "engine": "numpy",
}  # parabolic, numpy
par4 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 8e-3,
    "pxmax": 7e-3,
    "kind": "parabolic",
    "engine": "numba",
}  # parabolic, numba


def test_unknown_engine2D():
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(NotImplementedError):
        _ = FourierRadon2D(None, None, None, None, engine="foo")


def test_unknown_engine3D():
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(NotImplementedError):
        _ = FourierRadon3D(None, None, None, None, None, None, engine="foo")


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FourierRadon2D(par):
    """Dot-test and sparse inverse for FourierRadon2D operator"""
    dt, dh = 0.005, 1
    t = np.arange(par["nt"]) * dt
    h = np.arange(par["nhx"]) * dh
    px = np.linspace(0, par["pxmax"], par["npx"])
    nfft = int(2 ** np.ceil(np.log2(par["nt"])))

    x = np.zeros((par["npx"], par["nt"]))
    x[2, par["nt"] // 2] = 1

    Rop = FourierRadon2D(
        t,
        h,
        px,
        nfft,
        kind=par["kind"],
        engine=par["engine"],
        dtype="float64",
    )
    assert dottest(Rop, par["nhx"] * par["nt"], par["npx"] * par["nt"], rtol=1e-3)

    y = Rop * x.ravel()

    if par["engine"] == "numba":  # as numpy is too slow here...
        xinv, _, _ = fista(Rop, y, niter=200, eps=3e0)
        assert_array_almost_equal(x.ravel(), xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FourierRadon3D(par):
    """Dot-test and sparse inverse for FourierRadon3D operator"""
    dt, dhy, dhx = 0.005, 1, 1
    t = np.arange(par["nt"]) * dt
    hy = np.arange(par["nhy"]) * dhy
    hx = np.arange(par["nhx"]) * dhx
    py = np.linspace(0, par["pymax"], par["npy"])
    px = np.linspace(0, par["pxmax"], par["npx"])
    nfft = int(2 ** np.ceil(np.log2(par["nt"])))

    x = np.zeros((par["npy"], par["npx"], par["nt"]))
    x[3, 2, par["nt"] // 2] = 1

    Rop = FourierRadon3D(
        t,
        hy,
        hx,
        py,
        px,
        nfft,
        kind=(par["kind"], par["kind"]),
        engine=par["engine"],
        dtype="float64",
    )
    assert dottest(
        Rop,
        par["nhy"] * par["nhx"] * par["nt"],
        par["npy"] * par["npx"] * par["nt"],
        rtol=1e-3,
    )

    y = Rop * x.ravel()

    if par["engine"] == "numba":  # as numpy is too slow here...
        xinv, _, _ = fista(Rop, y, niter=200, eps=1e1)
        assert_array_almost_equal(x.ravel(), xinv, decimal=1)
