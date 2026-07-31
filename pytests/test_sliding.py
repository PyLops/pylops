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

from pylops.basicoperators import Identity, MatrixMult
from pylops.optimization.basic import cgls
from pylops.signalprocessing import Sliding1D, Sliding2D, Sliding3D
from pylops.signalprocessing.sliding1d import sliding1d_design, sliding1d_pad_to_next
from pylops.signalprocessing.sliding2d import sliding2d_design, sliding2d_pad_to_next
from pylops.signalprocessing.sliding3d import sliding3d_design, sliding3d_pad_to_next
from pylops.utils import dottest

par1 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 5,
    "novery": 0,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 5,
    "noverx": 0,
    # "winsx": 2,
    "tapertype": None,
    "savetaper": True,
}  # no overlap, no taper
par2 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 5,
    "novery": 0,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 5,
    "noverx": 0,
    # "winsx": 2,
    "tapertype": "hanning",
    "savetaper": True,
}  # no overlap, with taper
par3 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 4,
    "noverx": 2,
    # "winsx": 4,
    "tapertype": None,
    "savetaper": True,
}  # overlap, no taper
par4 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 4,
    "noverx": 2,
    # "winsx": 4,
    "tapertype": None,
    "savetaper": False,
}  # overlap, no taper (non saved)
par5 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 4,
    "noverx": 2,
    # "winsx": 4,
    "tapertype": "hanning",
    "savetaper": True,
}  # overlap, with taper
par6 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    # "winsy": 3,
    "npx": 10,
    "nwinx": 4,
    "noverx": 2,
    # "winsx": 4,
    "tapertype": "hanning",
    "savetaper": False,
}  # overlap, with taper (non saved)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_sliding1d_pad_to_next(par):
    """Check pad_to_next returns padded input that is fully covered
    by sliding windows"""
    for pad in range(0, 20):
        inpt = np.ones(par["npy"] + pad)
        inpt_pad, _, _, _, dwin_inends = sliding1d_pad_to_next(
            inpt, par["nwiny"], par["novery"], par["ny"]
        )
        assert inpt_pad.size == dwin_inends[1][-1]


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_sliding2d_pad_to_next(par):
    """Check pad_to_next returns padded input that is fully covered
    by sliding windows"""
    for pad in range(0, 20):
        inpt = np.ones((par["npy"] + pad, par["npx"]))
        inpt_pad, _, _, _, dwin_inends = sliding2d_pad_to_next(
            inpt, par["nwiny"], par["novery"], (par["ny"], par["nx"])
        )
        assert inpt_pad.shape[0] == dwin_inends[1][-1]


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_sliding3d_pad_to_next(par):
    """Check pad_to_next returns padded input that is fully covered
    by sliding windows"""
    for pad0 in range(0, 20):
        for pad1 in range(0, 20):
            inpt = np.ones((par["npy"] + pad0, par["npx"] + pad1, par["npx"]))
            inpt_pad, _, _, _, dwin_inends = sliding3d_pad_to_next(
                inpt,
                (par["nwiny"], par["nwinx"]),
                (par["novery"], par["noverx"]),
                (par["ny"], par["nx"], par["nx"]),
            )
            assert inpt_pad.shape[0] == dwin_inends[0][1][-1]
            assert inpt_pad.shape[1] == dwin_inends[1][1][-1]


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding1D(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding1D operator"""
    Op = MatrixMult(np.ones((par["nwiny"], par["ny"]), dtype=dtype), dtype=dtype)

    nwins, dim = sliding1d_design(par["npy"], par["nwiny"], par["novery"], par["ny"])[
        :2
    ]

    Slid = Sliding1D(
        Op,
        dim=dim,
        dimd=par["npy"],
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"],
        par["ny"] * nwins,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((nwins, par["ny"]), dtype=dtype)
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding1D_simOp(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding1D operator with
    Op applied to all windows simultaneously
    """
    nwins, dim = sliding1d_design(
        par["npy"], par["nwiny"], par["novery"], par["nwiny"]
    )[:2]

    Op = Identity((nwins, par["nwiny"]), dtype=dtype)

    Slid = Sliding1D(
        Op,
        dim=dim,
        dimd=par["npy"],
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"],
        par["nwiny"] * nwins,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )
    x = np.ones((nwins, par["nwiny"]), dtype=dtype)
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding2D(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding2D operator"""
    Op = MatrixMult(
        np.ones((par["nwiny"] * par["nt"], par["ny"] * par["nt"]), dtype=dtype),
        dtype=dtype,
    )

    nwins, dims = sliding2d_design(
        (par["npy"], par["nt"]), par["nwiny"], par["novery"], (par["ny"], par["nt"])
    )[:2]
    Slid = Sliding2D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["nt"]),
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"] * par["nt"],
        par["ny"] * par["nt"] * nwins,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones((nwins, par["ny"], par["nt"]), dtype=dtype)
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding2D_simOp(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding2D operator with
    Op applied to all windows simultaneously
    """
    nwins, dims = sliding2d_design(
        (par["npy"], par["nt"]), par["nwiny"], par["novery"], (par["nwiny"], par["nt"])
    )[:2]

    Op = Identity((nwins, par["nwiny"], par["nt"]), dtype=dtype)

    Slid = Sliding2D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["nt"]),
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"] * par["nt"],
        par["nwiny"] * par["nt"] * nwins,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((nwins, par["nwiny"], par["nt"]), dtype=dtype)
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding3D(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding3D operator"""
    Op = MatrixMult(
        np.ones(
            (
                par["nwiny"] * par["nwinx"] * par["nt"],
                par["ny"] * par["nx"] * par["nt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims = sliding3d_design(
        (par["npy"], par["npx"], par["nt"]),
        (par["nwiny"], par["nwinx"]),
        (par["novery"], par["noverx"]),
        (par["ny"], par["nx"], par["nt"]),
    )[:2]

    Slid = Sliding3D(
        Op,
        dims=dims,  # (par["ny"] * par["winsy"], par["nx"] * par["winsx"], par["nt"]),
        dimsd=(par["npy"], par["npx"], par["nt"]),
        nwin=(par["nwiny"], par["nwinx"]),
        nover=(par["novery"], par["noverx"]),
        nop=(par["ny"], par["nx"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"] * par["npx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((nwins[0], nwins[1], par["ny"], par["nx"], par["nt"]), dtype=dtype)
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Sliding3D_simOp(par, dtype):
    """Dot-test and forward/adjoint/inverse for Sliding3D operator with
    Op applied to all windows simultaneously
    """
    nwins, dims = sliding3d_design(
        (par["npy"], par["npx"], par["nt"]),
        (par["nwiny"], par["nwinx"]),
        (par["novery"], par["noverx"]),
        (par["nwiny"], par["nwinx"], par["nt"]),
    )[:2]

    Op = Identity((*nwins, par["nwiny"], par["nwinx"], par["nt"]), dtype=dtype)

    Slid = Sliding3D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npx"], par["nt"]),
        nwin=(par["nwiny"], par["nwinx"]),
        nover=(par["novery"], par["noverx"]),
        nop=(par["ny"], par["nx"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Slid,
        par["npy"] * par["npx"] * par["nt"],
        par["nwiny"] * par["nwinx"] * par["nt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    x = np.ones(
        (nwins[0], nwins[1], par["nwiny"], par["nwinx"], par["nt"]), dtype=dtype
    )
    y = Slid * x.ravel()
    xadj = Slid.H * y
    xinv = cgls(Slid, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)
