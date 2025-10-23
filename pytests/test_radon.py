import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.optimization.sparsity import fista
from pylops.signalprocessing import Radon2D, Radon3D
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


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_unknown_engine():
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(KeyError):
        _ = Radon2D(None, None, None, engine="foo")
    with pytest.raises(KeyError):
        _ = Radon3D(None, None, None, None, None, engine="foo")


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_Radon2D(par):
    """Dot-test, forward and adjoint consistency check
    (for onthefly parameter), and sparse inverse for Radon2D operator
    """
    dt, dh = 0.005, 1
    t = np.arange(par["nt"]) * dt
    h = np.arange(par["nhx"]) * dh
    px = np.linspace(0, par["pxmax"], par["npx"])
    x = np.zeros((par["npx"], par["nt"]))
    x[2, par["nt"] // 2] = 1

    Rop = Radon2D(
        t,
        h,
        px,
        centeredh=par["centeredh"],
        interp=par["interp"],
        kind=par["kind"],
        onthefly=False,
        engine=par["engine"],
        dtype="float64",
    )
    R1op = Radon2D(
        t,
        h,
        px,
        centeredh=par["centeredh"],
        interp=par["interp"],
        kind=par["kind"],
        onthefly=True,
        engine=par["engine"],
        dtype="float64",
    )
    assert dottest(Rop, par["nhx"] * par["nt"], par["npx"] * par["nt"], rtol=1e-3)

    y = Rop * x.ravel()
    y1 = R1op * x.ravel()
    assert_array_almost_equal(y, y1, decimal=4)

    xadj = Rop.H * y
    xadj1 = R1op.H * y
    assert_array_almost_equal(xadj, xadj1, decimal=4)

    xinv, _, _ = fista(Rop, y, niter=30, eps=1e0)
    assert_array_almost_equal(x.ravel(), xinv, decimal=1)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_Radon3D(par):
    """Dot-test, forward and adjoint consistency check
    (for onthefly parameter), and sparse inverse for Radon3D operator
    """
    dt, dhy, dhx = 0.005, 1, 1
    t = np.arange(par["nt"]) * dt
    hy = np.arange(par["nhy"]) * dhy
    hx = np.arange(par["nhx"]) * dhx
    py = np.linspace(0, par["pymax"], par["npy"])
    px = np.linspace(0, par["pxmax"], par["npx"])
    x = np.zeros((par["npy"], par["npx"], par["nt"]))
    x[3, 2, par["nt"] // 2] = 1

    Rop = Radon3D(
        t,
        hy,
        hx,
        py,
        px,
        centeredh=par["centeredh"],
        interp=par["interp"],
        kind=par["kind"],
        onthefly=False,
        engine=par["engine"],
        dtype="float64",
    )
    R1op = Radon3D(
        t,
        hy,
        hx,
        py,
        px,
        centeredh=par["centeredh"],
        interp=par["interp"],
        kind=par["kind"],
        onthefly=True,
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
    y1 = R1op * x.ravel()
    assert_array_almost_equal(y, y1, decimal=4)

    xadj = Rop.H * y
    xadj1 = R1op.H * y
    assert_array_almost_equal(xadj, xadj1, decimal=4)

    if Rop.engine == "numba":  # as numpy is too slow here...
        xinv, _, _ = fista(Rop, y, niter=200, eps=3e0)
        assert_array_almost_equal(x.ravel(), xinv, decimal=1)
