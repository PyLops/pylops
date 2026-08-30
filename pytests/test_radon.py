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
    with pytest.raises(ValueError, match="`engine` must be numpy"):
        _ = Radon2D(None, None, None, engine="foo")

    with pytest.raises(ValueError, match="`engine` must be numpy"):
        _ = Radon3D(None, None, None, None, None, engine="foo")


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_Radon2D_unknown_kind():
    """Check error is raised if unknown (and non-callable) kind is passed"""
    t = np.arange(11, dtype=np.float64) * 0.005
    h = np.arange(21, dtype=np.float64)
    px = np.linspace(0, 2e-2, 21, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="Wrong kind of basis function"):
        _ = Radon2D(t, h, px, kind="foo")


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Radon2D(par, dtype):
    """Dot-test, forward and adjoint consistency check
    (for onthefly parameter), and sparse inverse for Radon2D operator
    """
    dt, dh = 0.005, 1
    t = np.arange(par["nt"], dtype=dtype) * dt
    h = np.arange(par["nhx"], dtype=dtype) * dh
    px = np.linspace(0, par["pxmax"], par["npx"], dtype=dtype)
    x = np.zeros((par["npx"], par["nt"]), dtype=dtype)
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
        dtype=dtype,
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
        dtype=dtype,
    )
    assert dottest(
        Rop,
        par["nhx"] * par["nt"],
        par["npx"] * par["nt"],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    y = Rop * x.ravel()
    y1 = R1op * x.ravel()
    assert y.dtype == dtype
    assert y1.dtype == dtype
    assert_array_almost_equal(y, y1, decimal=4)

    xadj = Rop.H * y
    xadj1 = R1op.H * y
    assert xadj.dtype == dtype
    assert xadj1.dtype == dtype
    assert_array_almost_equal(xadj, xadj1, decimal=4)

    xinv, _, _ = fista(Rop, y, niter=30, eps=1e0)
    assert_array_almost_equal(x.ravel(), xinv, decimal=1)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_Radon2D_callable_kind():
    """Dot-test for Radon2D operator when kind is a custom callable"""
    dt, dh = 0.005, 1
    t = np.arange(par1["nt"], dtype=np.float64) * dt
    h = np.arange(par1["nhx"], dtype=np.float64) * dh
    px = np.linspace(0, par1["pxmax"], par1["npx"], dtype=np.float64)

    def _linear(x, t, px):
        return t + px * x

    Rop = Radon2D(
        t,
        h,
        px,
        centeredh=par1["centeredh"],
        interp=par1["interp"],
        kind=_linear,
        onthefly=False,
        engine="numpy",
        dtype=np.float64,
    )
    assert dottest(Rop, par1["nhx"] * par1["nt"], par1["npx"] * par1["nt"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Radon3D(par, dtype):
    """Dot-test, forward and adjoint consistency check
    (for onthefly parameter), and sparse inverse for Radon3D operator
    """
    dt, dhy, dhx = 0.005, 1, 1
    t = np.arange(par["nt"], dtype=dtype) * dt
    hy = np.arange(par["nhy"], dtype=dtype) * dhy
    hx = np.arange(par["nhx"], dtype=dtype) * dhx
    py = np.linspace(0, par["pymax"], par["npy"], dtype=dtype)
    px = np.linspace(0, par["pxmax"], par["npx"], dtype=dtype)
    x = np.zeros((par["npy"], par["npx"], par["nt"]), dtype=dtype)
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
        dtype=dtype,
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
        dtype=dtype,
    )
    assert dottest(
        Rop,
        par["nhy"] * par["nhx"] * par["nt"],
        par["npy"] * par["npx"] * par["nt"],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    y = Rop * x.ravel()
    y1 = R1op * x.ravel()
    assert y.dtype == dtype
    assert y1.dtype == dtype
    assert_array_almost_equal(y, y1, decimal=4)

    xadj = Rop.H * y
    xadj1 = R1op.H * y
    assert xadj.dtype == dtype
    assert xadj1.dtype == dtype
    assert_array_almost_equal(xadj, xadj1, decimal=4)

    if Rop.engine == "numba":  # as numpy is too slow here...
        xinv, _, _ = fista(Rop, y, niter=200, eps=3e0)
        assert_array_almost_equal(x.ravel(), xinv, decimal=1)
