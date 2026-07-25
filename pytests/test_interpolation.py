import os
from dataclasses import dataclass
from typing import Final, Literal

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import pytest

from pylops.signalprocessing import Bilinear, Interp
from pylops.utils import dottest


@dataclass(slots=True)
class InterpolationTestParameters:
    """
    Test parameters for testing :class:`pylops.signalprocessing.Interp`.

    """

    y_num: int
    x_num: int
    t_num: int
    imag: complex
    kind: Literal["nearest", "linear", "sinc", "cubic_spline"]


par1 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=0,
        kind="nearest",
    ),
    id="nearest - real",
)
par2 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=1j,
        kind="nearest",
    ),
    id="nearest - complex",
)
par3 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=0,
        kind="linear",
    ),
    id="linear - real",
)
par4 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=1j,
        kind="linear",
    ),
    id="linear - complex",
)
par5 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=0,
        kind="sinc",
    ),
    id="sinc - real",
)
par6 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=1j,
        kind="sinc",
    ),
    id="sinc - complex",
)
par7 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=0,
        kind="cubic_spline",
    ),
    id="cubic natural spline - real",
)
par8 = pytest.param(
    InterpolationTestParameters(
        y_num=21,
        x_num=11,
        t_num=20,
        imag=1j,
        kind="cubic_spline",
    ),
    id="cubic natural spline - complex",
)

# subsampling factor
SUBSAMPLING_PERCENTAGE: Final[float] = 0.4


def test_sincinterp():
    """Check accuracy of sinc interpolation of subsampled version of input
    signal
    """
    nt = 81
    dt = 0.004
    t = np.arange(nt) * dt

    ntsub = 10
    dtsub = dt / ntsub
    tsub = np.arange(nt * ntsub) * dtsub
    tsub = tsub[: np.where(tsub == t[-1])[0][0]]
    tsub += dt / 4
    x = (
        np.sin(2 * np.pi * 10 * t)
        + 0.4 * np.sin(2 * np.pi * 20 * t)
        - 2 * np.sin(2 * np.pi * 5 * t)
    )
    xsub = (
        np.sin(2 * np.pi * 10 * tsub)
        + 0.4 * np.sin(2 * np.pi * 20 * tsub)
        - 2 * np.sin(2 * np.pi * 5 * tsub)
    )

    iava = tsub[20:-20] / (dtsub * ntsub)  # exclude edges
    for tol in [None, 1e-2]:
        SI1op, iava = Interp(nt, iava, kind="sinc", tol=tol, dtype="float64")
        y = SI1op * x
        assert_array_almost_equal(xsub[20:-20], y, decimal=1)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par7),
        (par8),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Interp_1dsignal(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward for Interp operator for 1d signal"""
    if par.kind == "cubic_spline":
        pytest.skip("cubic_spline does not support CuPy arrays")

    np.random.seed(1)
    dtype = dtype if par.kind != "cubic_spline" else np.float64
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    x = np.random.normal(0, 1, par.x_num).astype(dtype) + par.imag * np.random.normal(
        0, 1, par.x_num
    ).astype(dtype)

    Nsub = int(np.round(par.x_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.x_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(par.x_num, iava, kind=par.kind, dtype=dtype1)
    assert dottest(
        Iop,
        Nsub,
        par.x_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(par.x_num, iava + 0.3, kind=par.kind, dtype=dtype1)
    assert dottest(
        Iop,
        Nsub,
        par.x_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # repeated indices
    with pytest.raises(ValueError, match="repeated"):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(par.x_num, iava_rep + 0.3, kind=par.kind, dtype=dtype)

    # forward
    y = Iop * x
    ydec = Idecop * x
    assert y.dtype == dtype1
    assert ydec.dtype == dtype1

    assert_array_almost_equal(y, x[iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[iava])


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par7),
        (par8),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Interp_2dsignal(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward for Restriction operator for 2d signal"""
    if par.kind == "cubic_spline":
        pytest.skip("cubic_spline does not support CuPy arrays")

    np.random.seed(1)
    dtype = dtype if par.kind != "cubic_spline" else np.float64
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    x = np.random.normal(0, 1, (par.x_num, par.t_num)).astype(
        dtype
    ) + par.imag * np.random.normal(0, 1, (par.x_num, par.t_num)).astype(dtype)

    # 1st direction
    Nsub = int(np.round(par.x_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.x_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(
        (par.x_num, par.t_num),
        iava,
        axis=0,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Iop,
        Nsub * par.t_num,
        par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(
        (par.x_num, par.t_num),
        iava + 0.3,
        axis=0,
        kind=par.kind,
        dtype=dtype1,
    )

    # repeated indices
    with pytest.raises(ValueError, match="repeated"):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(
            (par.x_num, par.t_num),
            iava_rep + 0.3,
            axis=0,
            kind=par.kind,
            dtype=dtype1,
        )

    y = (Iop * x.ravel()).reshape(Nsub, par.t_num)
    ydec = (Idecop * x.ravel()).reshape(Nsub, par.t_num)

    assert_array_almost_equal(y, x[iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[iava])

    # 2nd direction
    Nsub = int(np.round(par.t_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.t_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(
        (par.x_num, par.t_num),
        iava,
        axis=1,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Iop,
        par.x_num * Nsub,
        par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(
        (par.x_num, par.t_num),
        iava + 0.3,
        axis=1,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Idecop,
        par.x_num * Nsub,
        par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Iop * x.ravel()).reshape(par.x_num, Nsub)
    ydec = (Idecop * x.ravel()).reshape(par.x_num, Nsub)

    assert_array_almost_equal(y, x[:, iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[:, iava])


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par5),
        (par6),
        (par7),
        (par8),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Interp_3dsignal(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward  for Interp operator for 3d signal"""
    if par.kind == "cubic_spline":
        pytest.skip("cubic_spline does not support CuPy arrays")

    np.random.seed(1)
    dtype = dtype if par.kind != "cubic_spline" else np.float64
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    x = np.random.normal(0, 1, (par.y_num, par.x_num, par.t_num)).astype(
        dtype
    ) + par.imag * np.random.normal(0, 1, (par.y_num, par.x_num, par.t_num)).astype(
        dtype
    )

    # 1st direction
    Nsub = int(np.round(par.y_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.y_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava,
        axis=0,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Iop,
        Nsub * par.x_num * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava + 0.3,
        axis=0,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Idecop,
        Nsub * par.x_num * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # repeated indices
    with pytest.raises(ValueError, match="repeated"):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(
            (par.y_num, par.x_num, par.t_num),
            iava_rep + 0.3,
            axis=0,
            kind=par.kind,
            dtype=dtype,
        )

    y = (Iop * x.ravel()).reshape(Nsub, par.x_num, par.t_num)
    ydec = (Idecop * x.ravel()).reshape(Nsub, par.x_num, par.t_num)

    assert_array_almost_equal(y, x[iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[iava])

    # 2nd direction
    Nsub = int(np.round(par.x_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.x_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava,
        axis=1,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Iop,
        par.y_num * Nsub * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava + 0.3,
        axis=1,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Idecop,
        par.y_num * Nsub * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Iop * x.ravel()).reshape(par.y_num, Nsub, par.t_num)
    ydec = (Idecop * x.ravel()).reshape(par.y_num, Nsub, par.t_num)

    assert_array_almost_equal(y, x[:, iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[:, iava])

    # 3rd direction
    Nsub = int(np.round(par.t_num * SUBSAMPLING_PERCENTAGE))
    iava = np.sort(np.random.permutation(np.arange(par.t_num))[:Nsub])

    # fixed indices
    Iop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava,
        axis=2,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Iop,
        par.y_num * par.x_num * Nsub,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Idecop, _ = Interp(
        (par.y_num, par.x_num, par.t_num),
        iava + 0.3,
        axis=2,
        kind=par.kind,
        dtype=dtype1,
    )
    assert dottest(
        Idecop,
        par.y_num * par.x_num * Nsub,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Iop * x.ravel()).reshape(par.y_num, par.x_num, Nsub)
    ydec = (Idecop * x.ravel()).reshape(par.y_num, par.x_num, Nsub)

    assert_array_almost_equal(y, x[:, :, iava])
    if par.kind == "nearest":
        assert_array_almost_equal(ydec, x[:, :, iava])


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Bilinear_2dsignal(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward for Interp operator for 2d signal"""
    if par.imag == 1j:
        pytest.skip("cupy.add.at currently does not support complex numbers")

    np.random.seed(1)
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    x = np.random.normal(0, 1, (par.x_num, par.t_num)).astype(
        dtype1
    ) + par.imag * np.random.normal(0, 1, (par.x_num, par.t_num)).astype(dtype1)

    # fixed indices
    iava = np.vstack((np.arange(0, 10), np.arange(0, 10)))
    Iop = Bilinear(iava, dims=(par.x_num, par.t_num), dtype=dtype1)
    assert dottest(
        Iop,
        10,
        par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Nsub = int(np.round(par.x_num * par.t_num * SUBSAMPLING_PERCENTAGE))
    iavadec = np.vstack(
        (
            np.random.uniform(0, par.x_num - 1, Nsub),
            np.random.uniform(0, par.t_num - 1, Nsub),
        )
    )
    Idecop = Bilinear(iavadec, dims=(par.x_num, par.t_num), dtype=dtype1)
    assert dottest(
        Idecop,
        Nsub,
        par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # repeated indices
    with pytest.raises(ValueError, match="repeated"):
        iava_rep = iava.copy()
        iava_rep[::, -1] = iava_rep[::, 0]
        _, _ = Bilinear(iava_rep, dims=(par.x_num, par.t_num), dtype=dtype1)

    y = Iop * x.ravel()
    assert_array_almost_equal(y, x[iava[0], iava[1]])


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Bilinear_2dsignal_flatten(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward for Interp operator for 2d signal with forceflat"""
    if par.imag == 1j:
        pytest.skip("cupy.add.at currently does not support complex numbers")

    np.random.seed(1)
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    dims = (par.x_num, par.t_num)
    flat_dims = par.x_num * par.t_num
    dimsd = 10

    x = np.random.normal(0, 1, dims).astype(dtype1) + par.imag * np.random.normal(
        0, 1, dims
    ).astype(dtype1)

    iava = np.vstack((np.arange(0, dimsd), np.arange(0, dimsd)))
    Iop_True = Bilinear(iava, dims=dims, dtype=dtype1, forceflat=True)
    y = Iop_True @ x
    xadj = Iop_True.H @ y
    assert y.shape == (dimsd,)
    assert xadj.shape == (flat_dims,)

    Iop_None = Bilinear(iava, dims=dims, dtype=dtype1)
    y = Iop_None @ x
    xadj = Iop_None.H @ y
    assert y.shape == (dimsd,)
    assert xadj.shape == dims


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Bilinear_3dsignal(par: InterpolationTestParameters, dtype: np.dtype):
    """Dot-test and forward for Interp operator for 3d signal"""
    if par.imag == 1j:
        pytest.skip("cupy.add.at currently does not support complex numbers")
    if par.kind == "nearest":
        pytest.skip("nearest does not support CuPy arrays")

    np.random.seed(1)
    dtype1 = (np.empty(0, dtype=dtype) + par.imag * np.empty(0, dtype=dtype)).dtype

    x = np.random.normal(0, 1, (par.y_num, par.x_num, par.t_num)).astype(
        dtype1
    ) + par.imag * np.random.normal(0, 1, (par.y_num, par.x_num, par.t_num)).astype(
        dtype1
    )

    # fixed indices
    iava = np.vstack((np.arange(0, 10), np.arange(0, 10)))
    Iop = Bilinear(iava, dims=(par.y_num, par.x_num, par.t_num), dtype=dtype1)
    assert dottest(
        Iop,
        10 * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # decimal indices
    Nsub = int(np.round(par.y_num * par.t_num * SUBSAMPLING_PERCENTAGE))
    iavadec = np.vstack(
        (
            np.random.uniform(0, par.y_num - 1, Nsub),
            np.random.uniform(0, par.x_num - 1, Nsub),
        )
    )
    Idecop = Bilinear(iavadec, dims=(par.y_num, par.x_num, par.t_num), dtype=dtype1)
    assert dottest(
        Idecop,
        Nsub * par.t_num,
        par.y_num * par.x_num * par.t_num,
        complexflag=0 if par.imag == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # repeated indices
    with pytest.raises(ValueError, match="repeated"):
        iava_rep = iava.copy()
        iava_rep[::, -1] = iava_rep[::, 0]
        _, _ = Bilinear(iava_rep, dims=(par.y_num, par.x_num, par.t_num), dtype=dtype1)

    y = Iop * x.ravel()
    assert_array_almost_equal(y, x[iava[0], iava[1]].ravel())
