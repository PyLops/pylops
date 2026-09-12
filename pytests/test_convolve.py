import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal
    from cupyx.scipy.signal.windows import triang

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal
    from scipy.signal.windows import triang

    backend = "numpy"

import pytest

from pylops import VStack
from pylops.optimization.basic import lsqr
from pylops.signalprocessing import Convolve1D, Convolve2D, ConvolveND
from pylops.utils import dottest

# filters
nfilt = (5, 6, 5)
h1 = triang(nfilt[0], sym=True)
h2 = np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True))
h3 = np.outer(
    np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True)),
    triang(nfilt[2], sym=True),
).reshape(nfilt)

# convolution methods (``direct`` and ``fft`` are the only ones allowed for a
# one-dimensional model, ``fft`` and ``overlapadd`` for a multi-dimensional one)
methods = (None, "direct", "fft", "overlapadd")
methods_1d = (None, "direct", "fft")
methods_nd = (None, "fft", "overlapadd")


def _broadcast_filter(h, dims, axis):
    """Expand a 1d filter over the other dimensions of a multi-dimensional model,
    returning a filter with the same number of dimensions as the model"""
    shape = list(dims)
    shape[axis] = h.size
    hdims = np.ones(len(dims), dtype=int)
    hdims[axis] = h.size
    return (h.reshape(tuple(hdims)) * np.ones(tuple(shape), dtype=h.dtype)).astype(
        h.dtype
    )


def _apply_along_axis(Op, x, axis):
    """Apply a 1d operator to every 1d slice of ``x`` taken along ``axis``"""
    xm = np.moveaxis(x, axis, -1)
    shape = xm.shape
    xm = xm.reshape(-1, shape[-1])
    ym = np.stack([Op * xm[i] for i in range(xm.shape[0])])
    ym = ym.reshape(shape[:-1] + (ym.shape[-1],))
    return np.moveaxis(ym, -1, axis)


par1_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 0,
}  # zero phase, first direction
par2_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": 0,
    "axis": 0,
}  # non-zero phase, first direction
par3_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 1,
}  # zero phase, second direction
par4_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": nfilt[0] // 2 - 1,
    "axis": 1,
}  # non-zero phase, second direction
par5_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 2,
}  # zero phase, third direction
par6_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": nfilt[0] // 2 - 1,
    "axis": 2,
}  # non-zero phase, third direction

par1_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 0,
}  # zero phase, first direction
par2_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 0,
}  # non-zero phase, first direction
par3_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 1,
}  # zero phase, second direction
par4_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 1,
}  # non-zero phase, second direction
par5_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 2,
}  # zero phase, third direction
par6_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 2,
}  # non-zero phase, third direction

par1_3d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "nt": 5,
    "offset": (nfilt[0] // 2, nfilt[1] // 2, nfilt[2] // 2),
    "axis": 0,
}  # zero phase, all directions
par2_3d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "nt": 5,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1, nfilt[2] // 2 + 1),
    "axis": 0,
}  # non-zero phase, first direction


def test_Convolve1D_method_error():
    """Error raised by Convolve1D operator when an invalid method is chosen"""
    # overlapadd is not allowed for a one-dimensional model
    with pytest.raises(ValueError, match="`method` must be direct or fft"):
        Convolve1D(nfilt[0] * 4, h=h1, offset=nfilt[0] // 2, method="overlapadd")

    # direct is not allowed for a multi-dimensional model
    with pytest.raises(ValueError, match="`method` must be fft or overlapadd"):
        Convolve1D(
            (nfilt[0] * 4, nfilt[0] * 4), h=h1, offset=nfilt[0] // 2, method="direct"
        )


@pytest.mark.parametrize(
    "par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d), (par5_1d), (par6_1d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", methods)
def test_Convolve1D_short(par, dtype, method):
    """Dot-test and inversion for Convolve1D operator with short filter"""
    np.random.seed(10)
    # 1D
    if par["axis"] == 0 and method in methods_1d:
        Cop = Convolve1D(
            par["nx"],
            h=h1.astype(dtype),
            offset=par["offset"],
            method=method,
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["nx"],
            par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros(par["nx"], dtype=dtype)
        x[par["nx"] // 2] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x,
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 2D
    if par["axis"] < 2 and method in methods_nd:
        Cop = Convolve1D(
            (par["ny"], par["nx"]),
            h=h1.astype(dtype),
            offset=par["offset"],
            axis=par["axis"],
            method=method,
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        x[
            int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
            int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        ] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x.ravel(),
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 3D
    if method in methods_nd:
        Cop = Convolve1D(
            (par["nz"], par["ny"], par["nx"]),
            h=h1.astype(dtype),
            offset=par["offset"],
            axis=par["axis"],
            method=method,
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["nz"] * par["ny"] * par["nx"],
            par["nz"] * par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
        x[
            int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
            int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
            int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        ] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x.ravel(),
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize(
    "par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d), (par5_1d), (par6_1d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", methods_1d)
def test_Convolve1D_long(par, dtype, method):
    """Dot-test and inversion for Convolve1D operator with long filter"""
    np.random.seed(10)
    # 1D
    if par["axis"] == 0:
        x = np.zeros(par["nx"], dtype=dtype)
        x[par["nx"] // 2] = 1.0
        Xop = Convolve1D(
            nfilt[0], h=x, offset=nfilt[0] // 2, method=method, dtype=dtype
        )
        assert dottest(
            Xop,
            par["nx"],
            nfilt[0],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        # Forward and adjoint dtype check
        y = Xop * h1.astype(dtype)
        h1adj = Xop.H * y
        assert y.dtype == dtype
        assert h1adj.dtype == dtype

        # Inverse
        h1lsqr = lsqr(
            Xop, Xop * h1, damp=1e-20, niter=200, atol=1e-8, btol=1e-8, show=0
        )[0]
        assert_array_almost_equal(h1, h1lsqr, decimal=1)


@pytest.mark.parametrize(
    "par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d), (par5_1d), (par6_1d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", methods_nd)
@pytest.mark.parametrize("hlen", ["short", "long"])
@pytest.mark.parametrize("hndim", ["1d", "nd"])
@pytest.mark.parametrize("nheven", [nfilt[0] - 1, nfilt[0]])
def test_Convolve1D_nd(par, dtype, method, hlen, hndim, nheven):
    """Dot-test for Convolve1D operator applied to a multi-dimensional model,
    for both compact and extended filters provided either as a 1d array or with
    the same number of dimensions as the model. The result is compared against
    the equivalent 1d operator applied to each slice along ``axis``. Note that
    inversion is not tested here as it is already covered by
    ``test_Convolve1D_*`` and, for extended filters, the system is
    heavily overdetermined.
    """
    np.random.seed(10)

    def make_filter(dims, axis):
        nh = nheven if hlen == "short" else dims[axis] + nheven
        h = triang(nh, sym=True).astype(dtype)
        return h, (h if hndim == "1d" else _broadcast_filter(h, dims, axis))

    offset = min(par["offset"], nheven - 1)

    # 1D on 2D
    if par["axis"] < 2:
        dims = (par["ny"], par["nx"])
        h, hop = make_filter(dims, par["axis"])
        Cop = Convolve1D(
            dims,
            h=hop,
            offset=offset,
            axis=par["axis"],
            method=method,
            dtype=dtype,
        )
        assert dottest(
            Cop,
            np.prod(Cop.dimsd),
            np.prod(Cop.dims),
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.random.normal(0.0, 1.0, dims).astype(dtype)

        # Forward and adjoint dtype check
        y = (Cop * x).reshape(Cop.dimsd)
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Equivalence with the 1d operator applied slice by slice
        C1op = Convolve1D(
            dims[par["axis"]],
            h=h,
            offset=offset,
            method=None if method == "overlapadd" else method,
            dtype=dtype,
        )
        assert_array_almost_equal(y, _apply_along_axis(C1op, x, par["axis"]), decimal=3)
        assert_array_almost_equal(
            xadj.reshape(Cop.dims),
            _apply_along_axis(C1op.H, y, par["axis"]),
            decimal=3,
        )

    # 1D on 3D
    dims = (par["nz"], par["ny"], par["nx"])
    _, hop = make_filter(dims, par["axis"])
    Cop = Convolve1D(
        dims,
        h=hop,
        offset=offset,
        axis=par["axis"],
        method=method,
        dtype=dtype,
    )
    assert dottest(
        Cop,
        np.prod(Cop.dimsd),
        np.prod(Cop.dims),
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", methods_nd)
@pytest.mark.parametrize("hlen", ["short", "long"])
def test_Convolve1D_broadcast(par, dtype, method, hlen):
    """Dot-test and comparison with a VStack of 1d operators for Convolve1D
    applied to a model with a singleton dimension, which the filter broadcasts
    over (e.g. a single wavelet convolved with a set of traces)"""
    np.random.seed(10)
    nx, ntraces = par["nx"], 4
    nh = nfilt[0] if hlen == "short" else nx + nfilt[0]

    # One filter per trace, all different
    hs = np.vstack([triang(nh, sym=True) * (1.0 + i) for i in range(ntraces)]).astype(
        dtype
    )
    offset = par["offset"] if hlen == "short" else nx // 2

    Cop = Convolve1D((1, nx), h=hs, offset=offset, axis=-1, method=method, dtype=dtype)
    assert Cop.dims == (1, nx)
    assert tuple(Cop.dimsd) == (ntraces, nh if hlen == "long" else nx)
    assert dottest(
        Cop,
        np.prod(Cop.dimsd),
        np.prod(Cop.dims),
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Equivalence with a vertical stack of the corresponding 1d operators
    Vop = VStack(
        [
            Convolve1D(
                nx,
                h=hs[i],
                offset=offset,
                method=None if method == "overlapadd" else method,
                dtype=dtype,
            )
            for i in range(ntraces)
        ]
    )
    x = np.random.normal(0.0, 1.0, nx).astype(dtype)
    y = np.random.normal(0.0, 1.0, int(np.prod(Cop.dimsd))).astype(dtype)
    assert_array_almost_equal(Cop * x, Vop * x, decimal=3)
    assert_array_almost_equal(Cop.H * y, Vop.H * y, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1_2d), (par2_2d), (par3_2d), (par4_2d), (par5_2d), (par6_2d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve2D(par, dtype):
    """Dot-test and inversion for Convolve2D operator"""
    # 2D on 2D
    if par["axis"] == 2:
        Cop = Convolve2D(
            (par["ny"], par["nx"]),
            h=h2.astype(dtype),
            offset=par["offset"],
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        x[
            int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
            int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        ] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x.ravel(),
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2D on 3D
    axes = list(range(3))
    axes.remove(par["axis"])
    Cop = Convolve2D(
        (par["nz"], par["ny"], par["nx"]),
        h=h2.astype(dtype),
        offset=par["offset"],
        axes=axes,
        dtype="float64",
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0

    # Forward and adjoint dtype check
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # Inverse
    xlsqr = lsqr(
        Cop,
        Cop * x.ravel(),
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    # due to ringing in solution we cannot use assert_array_almost_equal
    assert np.linalg.norm(xlsqr - x) / np.linalg.norm(xlsqr) < 2e-1


@pytest.mark.parametrize("par", [(par1_3d), (par2_3d)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve3D(par, dtype):
    """Dot-test and inversion for ConvolveND operator"""
    # 3D on 3D
    Cop = ConvolveND(
        (par["nz"], par["ny"], par["nx"]),
        h=h3.astype(dtype),
        offset=par["offset"],
        dtype=dtype,
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0

    # Forward and adjoint dtype check
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # Inverse
    xlsqr = lsqr(
        Cop, y, x0=np.zeros_like(x), damp=1e-20, niter=400, atol=1e-8, btol=1e-8, show=0
    )[0]
    # due to ringing in solution we cannot use assert_array_almost_equal
    assert np.linalg.norm(xlsqr - x) / np.linalg.norm(xlsqr) < 2e-1

    # 3D on 4D (only modelling)
    Cop = ConvolveND(
        (par["nz"], par["ny"], par["nx"], par["nt"]),
        h=h3.astype(dtype),
        offset=par["offset"],
        axes=[0, 1, 2],
        dtype=dtype,
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"] * par["nt"],
        par["nz"] * par["ny"] * par["nx"] * par["nt"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward and adjoint dtype check
    x = np.zeros((par["nz"], par["ny"], par["nx"], par["nt"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nt"] / 2 - 3) : int(par["nt"] / 2 + 3),
    ] = 1.0

    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype
