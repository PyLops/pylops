import multiprocessing

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FourierRadon2D
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
}  # linear, centered, linear interp, numpy
par2 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "centeredh": False,
    "kind": "linear",
    "interp": True,
    "engine": "numpy",
}  # linear, uncentered, linear interp, numpy
par3 = {
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
    "engine": "numba",
}  # linear, centered, linear interp, numba
par4 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "centeredh": False,
    "kind": "linear",
    "interp": False,
    "engine": "numba",
}  # linear, uncentered, linear interp, numba
par5 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 8e-3,
    "pxmax": 7e-3,
    "centeredh": True,
    "kind": "parabolic",
    "interp": False,
    "engine": "numpy",
}  # parabolic, centered, no interp, numpy
par6 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 8e-3,
    "pxmax": 7e-3,
    "centeredh": False,
    "kind": "parabolic",
    "interp": True,
    "engine": "numba",
}  # parabolic, uncentered, interp, numba
par7 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 9e-2,
    "pxmax": 8e-2,
    "centeredh": True,
    "kind": "hyperbolic",
    "interp": True,
    "engine": "numpy",
}  # hyperbolic, centered, interp, numpy
par8 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 10,
    "npx": 21,
    "npy": 17,
    "pymax": 7e-2,
    "pxmax": 8e-2,
    "centeredh": False,
    "kind": "hyperbolic",
    "interp": False,
    "engine": "numba",
}  # hyperbolic, uncentered, interp, numba


def test_unknown_engine():
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(KeyError):
        _ = FourierRadon2D(None, None, None, None, engine="foo")


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
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
