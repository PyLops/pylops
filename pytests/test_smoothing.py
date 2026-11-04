import os
from math import prod

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import pytest

from pylops.basicoperators import Smoothing1D, Smoothing2D, SmoothingND
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {  # even, first direction
    "nt": 2,
    "nz": 10,
    "ny": 30,
    "nx": 20,
    "axis": 0,
    "nsmooth": (3, 5, 3),
}
par2 = {  # odd, first direction
    "nt": 3,
    "nz": 11,
    "ny": 51,
    "nx": 31,
    "axis": 0,
    "nsmooth": (3, 5, 3),
}
par3 = {"nz": 10, "ny": 30, "nx": 20, "axis": 1}  # even, second direction
par4 = {"nz": 11, "ny": 51, "nx": 31, "axis": 1}  # odd, second direction
par5 = {"nz": 10, "ny": 30, "nx": 20, "axis": 2}  # even, third direction
par6 = {"nz": 11, "ny": 51, "nx": 31, "axis": 2}  # odd, third direction

np.random.seed(0)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Smoothing1D(par, dtype):
    """Dot-test and forward/adjoint/inversion for Smoothing1D"""

    # 1d kernel on 1d signal
    D1op = Smoothing1D(nsmooth=5, dims=par["nx"], dtype=dtype)
    assert dottest(
        D1op,
        par["nx"],
        par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.random.normal(0, 1, par["nx"]).astype(dtype)
    y = D1op * x
    xadj = D1op.H * y
    xlsqr = lsqr(
        D1op,
        y,
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=100,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xlsqr, decimal=1 if dtype == np.float32 else 3)

    # 1d kernel on 2d signal
    D1op = Smoothing1D(
        nsmooth=5, dims=(par["ny"], par["nx"]), axis=par["axis"], dtype=dtype
    )
    assert dottest(
        D1op,
        par["ny"] * par["nx"],
        par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])).astype(dtype).ravel()
    y = D1op * x
    xadj = D1op.H * y
    xlsqr = lsqr(
        D1op,
        y,
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=100,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xlsqr, decimal=1 if dtype == np.float32 else 3)

    # 1d kernel on 3d signal
    D1op = Smoothing1D(
        nsmooth=5,
        dims=(par["nz"], par["ny"], par["nx"]),
        axis=par["axis"],
        dtype=dtype,
    )
    assert dottest(
        D1op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
    )

    x = np.random.normal(0, 1, (par["nz"], par["ny"], par["nx"])).astype(dtype).ravel()
    y = D1op * x
    xadj = D1op.H * y
    xlsqr = lsqr(
        D1op,
        y,
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xlsqr, decimal=1 if dtype == np.float32 else 3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Smoothing2D(par, dtype):
    """Dot-test for Smoothing2D"""
    # 2d kernel on 2d signal
    if par["axis"] < 2:
        D2op = Smoothing2D(nsmooth=(5, 5), dims=(par["ny"], par["nx"]), dtype=dtype)
        assert dottest(
            D2op,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        # forward
        x = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        x[par["ny"] // 2, par["nx"] // 2] = 1.0
        x = x.ravel()
        y = D2op * x
        xadj = D2op.H * y
        y = y.reshape(par["ny"], par["nx"])
        assert y.dtype == dtype
        assert xadj.dtype == dtype
        assert_array_almost_equal(
            y[par["ny"] // 2 - 2 : par["ny"] // 2 + 3 :, par["nx"] // 2],
            np.ones(5) / 25,
        )
        assert_array_almost_equal(
            y[par["ny"] // 2, par["nx"] // 2 - 2 : par["nx"] // 2 + 3], np.ones(5) / 25
        )
        # inverse
        xlsqr = lsqr(
            D2op,
            y.ravel(),
            x0=np.zeros_like(x),
            damp=1e-10,
            niter=400,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2d kernel on 3d signal
    axes = list(range(3))
    axes.remove(par["axis"])
    D2op = Smoothing2D(
        nsmooth=(5, 5),
        dims=(par["nz"], par["ny"], par["nx"]),
        axes=axes,
        dtype=dtype,
    )
    assert dottest(
        D2op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # forward
    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2] = 1.0
    x = x.ravel()
    y = D2op * x
    xadj = D2op.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    y = y.reshape(par["nz"], par["ny"], par["nx"])
    if par["axis"] == 0:
        assert_array_almost_equal(
            y[
                par["nz"] // 2,
                par["ny"] // 2 - 2 : par["ny"] // 2 + 3,
                par["nx"] // 2 - 2 : par["nx"] // 2 + 3,
            ],
            np.ones((5, 5)) / 25,
        )
    elif par["axis"] == 1:
        assert_array_almost_equal(
            y[
                par["nz"] // 2 - 2 : par["nz"] // 2 + 3,
                par["ny"] // 2,
                par["nx"] // 2 - 2 : par["nx"] // 2 + 3,
            ],
            np.ones((5, 5)) / 25,
        )
    elif par["axis"] == 2:
        assert_array_almost_equal(
            y[
                par["nz"] // 2 - 2 : par["nz"] // 2 + 3,
                par["ny"] // 2 - 2 : par["ny"] // 2 + 3,
                par["nx"] // 2,
            ],
            np.ones((5, 5)) / 25,
        )

    # inverse
    xlsqr = lsqr(
        D2op,
        y.ravel(),
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=400,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SmoothingND(par, dtype):
    """Dot-test for SmoothingND"""
    # 3d signal
    axes = list(range(3))
    D3op = SmoothingND(
        nsmooth=(3, 3, 3),
        dims=(par["nz"], par["ny"], par["nx"]),
        axes=axes,
        dtype=dtype,
    )
    assert dottest(
        D3op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # forward
    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2] = 1.0
    x = x.ravel()
    y = D3op * x
    xadj = D3op.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    y = y.reshape(par["nz"], par["ny"], par["nx"])
    assert_array_almost_equal(
        y[
            par["nz"] // 2 - 1 : par["nz"] // 2 + 2,
            par["ny"] // 2 - 1 : par["ny"] // 2 + 2,
            par["nx"] // 2 - 1 : par["nx"] // 2 + 2,
        ],
        np.ones((3, 3, 3)) / 27,
    )
    assert_array_almost_equal(
        y[par["nz"] // 2, par["ny"] // 2 - 1 : par["ny"] // 2 + 2, par["nx"] // 2],
        np.ones(3) / 27,
    )
    assert_array_almost_equal(
        y[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2 - 1 : par["nx"] // 2 + 2],
        np.ones(3) / 27,
    )

    # inverse
    xlsqr = lsqr(
        D3op,
        y.ravel(),
        x0=np.zeros_like(x),
        damp=1e-10,
        niter=400,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("axes", ["default", None, "permuted"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_SmoothingND_axes(par, axes, dtype):
    """Dot-test and analytic impulse response for default and explicit axes."""
    configurations = [
        # 2D on 2D
        ((par["ny"], par["nx"]), par["nsmooth"][:2]),
        # 3D on 3D
        ((par["nz"], par["ny"], par["nx"]), par["nsmooth"]),
        # 3D on 4D (only modelling)
        ((par["nt"], par["nz"], par["ny"], par["nx"]), par["nsmooth"]),
        # 2D on 3D (only modelling)
        ((par["nz"], par["ny"], par["nx"]), par["nsmooth"][:2]),
    ]
    for dims, nsmooth in configurations:
        physical_axes = (
            tuple((len(dims) - 1 + i) % len(dims) for i in range(len(nsmooth)))
            if axes == "permuted"
            else tuple(range(len(dims) - len(nsmooth), len(dims)))
        )
        kwargs = {} if axes == "default" else {"axes": None if axes is None else physical_axes}
        Sop = SmoothingND(nsmooth=nsmooth, dims=dims, dtype=dtype, **kwargs)
        assert dottest(
            Sop,
            prod(dims),
            prod(dims),
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        # Forward and adjoint dtype check
        center = tuple(d // 2 for d in dims)
        x = np.zeros(dims, dtype=dtype)
        x[center] = 1
        y = Sop * x
        xadj = Sop.H * x
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Analytic box response, clipped at the model boundaries
        expected = np.zeros_like(x)
        slices = [slice(c, c + 1) for c in center]
        for axis, length in zip(physical_axes, nsmooth, strict=True):
            slices[axis] = slice(
                max(0, center[axis] - length // 2),
                min(dims[axis], center[axis] + length // 2 + 1),
            )
        expected[tuple(slices)] = 1 / prod(nsmooth)
        assert_array_almost_equal(y, expected, decimal=5)
        assert_array_almost_equal(xadj, expected, decimal=5)
