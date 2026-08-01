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

from pylops.basicoperators import MatrixMult
from pylops.optimization.basic import cgls
from pylops.signalprocessing import Patch2D, Patch3D
from pylops.signalprocessing.patch2d import patch2d_design, patch2d_pad_to_next
from pylops.signalprocessing.patch3d import patch3d_design, patch3d_pad_to_next
from pylops.utils import dottest

par1 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 5,
    "novery": 0,
    # "winsy": 3,
    "npx": 13,
    "nwinx": 5,
    "noverx": 0,
    # "winsx": 2,
    "npt": 10,
    "nwint": 5,
    "novert": 0,
    # "winst": 2,
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
    "npx": 13,
    "nwinx": 5,
    "noverx": 0,
    # "winsx": 2,
    "npt": 10,
    "nwint": 5,
    "novert": 0,
    # "winst": 2,
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
    "npx": 13,
    "nwinx": 5,
    "noverx": 2,
    # "winsx": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    # "winst": 4,
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
    "npx": 13,
    "nwinx": 5,
    "noverx": 2,
    # "winsx": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    # "winst": 4,
    "tapertype": None,
    "savetaper": False,
}  # overlap, no taper (non saved
par5 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    # "winsy": 3,
    "npx": 13,
    "nwinx": 5,
    "noverx": 2,
    # "winsx": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    # "winst": 4,
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
    "npx": 13,
    "nwinx": 5,
    "noverx": 2,
    # "winsx": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    # "winst": 4,
    "tapertype": "hanning",
    "savetaper": False,
}  # overlap, with taper (non saved)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_patch2d_pad_to_next(par):
    """Check pad_to_next returns padded input that is fully covered
    by patches"""
    for pad0 in range(0, 20):
        for pad1 in range(0, 20):
            inpt = np.ones((par["npy"] + pad0, par["npx"] + pad1))
            inpt_pad, _, _, _, dwin_inends = patch2d_pad_to_next(
                inpt,
                (par["nwiny"], par["nwinx"]),
                (par["novery"], par["noverx"]),
                (par["ny"], par["nx"]),
            )
            assert inpt_pad.shape[0] == dwin_inends[0][1][-1]
            assert inpt_pad.shape[1] == dwin_inends[1][1][-1]


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_patch3d_pad_to_next(par):
    """Check pad_to_next returns padded input that is fully covered
    by patches"""
    for pad0 in range(0, 20):
        for pad1 in range(0, 20):
            for pad2 in range(0, 20):
                inpt = np.ones(
                    (par["npy"] + pad0, par["npx"] + pad1, par["npt"] + pad2)
                )
                inpt_pad, _, _, _, dwin_inends = patch3d_pad_to_next(
                    inpt,
                    (par["nwiny"], par["nwinx"], par["nwint"]),
                    (par["novery"], par["noverx"], par["novert"]),
                    (par["ny"], par["nx"], par["nt"]),
                )
                assert inpt_pad.shape[0] == dwin_inends[0][1][-1]
                assert inpt_pad.shape[1] == dwin_inends[1][1][-1]
                assert inpt_pad.shape[2] == dwin_inends[2][1][-1]


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch2D(par, dtype):
    """Dot-test and forward/adjoint/inverse for Patch2D operator"""
    Op = MatrixMult(
        np.ones((par["nwiny"] * par["nwint"], par["ny"] * par["nt"]), dtype=dtype),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch2d_design(
        (par["npy"], par["npt"]),
        (par["nwiny"], par["nwint"]),
        (par["novery"], par["novert"]),
        (par["ny"], par["nt"]),
    )
    Pop = Patch2D(
        Op,
        dims=dims,  # (par["ny"] * par["winsy"], par["nt"] * par["winst"]),
        dimsd=(par["npy"], par["npt"]),
        nwin=(par["nwiny"], par["nwint"]),
        nover=(par["novery"], par["novert"]),
        nop=(par["ny"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["ny"] * par["nt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones((nwins[0], nwins[1], par["ny"], par["nt"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch2D_scalings(par, dtype):
    """Dot-test and forward/adjoint/inverse for Patch2D operator with scalings"""
    Op = MatrixMult(
        np.ones((par["nwiny"] * par["nwint"], par["ny"] * par["nt"]), dtype=dtype),
        dtype=dtype,
    )
    scalings = np.arange(par["nwiny"] * par["nwint"], dtype=dtype) + 1.0

    nwins, dims, _, _ = patch2d_design(
        (par["npy"], par["npt"]),
        (par["nwiny"], par["nwint"]),
        (par["novery"], par["novert"]),
        (par["ny"], par["nt"]),
    )
    Pop = Patch2D(
        Op,
        dims=dims,  # (par["ny"] * par["winsy"], par["nt"] * par["winst"]),
        dimsd=(par["npy"], par["npt"]),
        nwin=(par["nwiny"], par["nwint"]),
        nover=(par["novery"], par["novert"]),
        nop=(par["ny"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
        scalings=scalings,
    )
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["ny"] * par["nt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones((nwins[0], nwins[1], par["ny"], par["nt"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch2D_singlepatch1(par, dtype):
    """Dot-test and inverse for Patch2D operator with single patch in first dimension"""
    Op = MatrixMult(
        np.ones((par["npy"] * par["nwint"], par["npy"] * par["nt"]), dtype=dtype),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch2d_design(
        (par["npy"], par["npt"]),
        (par["npy"], par["nwint"]),
        (0, par["novert"]),
        (par["npy"], par["nt"]),
    )

    Pop = Patch2D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npt"]),
        nwin=(par["npy"], par["nwint"]),
        nover=(0, par["novert"]),
        nop=(par["npy"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert nwins[0] == 1
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["npy"] * par["nt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones((nwins[0], nwins[1], par["npy"], par["nt"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch2D_singlepatch2(par, dtype):
    """Dot-test and inverse for Patch2D operator with single patch in second dimension"""
    Op = MatrixMult(
        np.ones((par["nwiny"] * par["npt"], par["ny"] * par["npt"]), dtype=dtype),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch2d_design(
        (par["npy"], par["npt"]),
        (par["nwiny"], par["npt"]),
        (par["novery"], 0),
        (par["ny"], par["npt"]),
    )
    Pop = Patch2D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npt"]),
        nwin=(par["nwiny"], par["npt"]),
        nover=(par["novery"], 0),
        nop=(par["ny"], par["npt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert nwins[1] == 1
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["ny"] * par["npt"] * nwins[0] * nwins[1],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones((nwins[0], nwins[1], par["ny"], par["nt"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch3D(par, dtype):
    """Dot-test and inverse for Patch3D operator"""
    Op = MatrixMult(
        np.ones(
            (
                par["nwiny"] * par["nwinx"] * par["nwint"],
                par["ny"] * par["nx"] * par["nt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch3d_design(
        (par["npy"], par["npx"], par["npt"]),
        (par["nwiny"], par["nwinx"], par["nwint"]),
        (par["novery"], par["noverx"], par["novert"]),
        (par["ny"], par["nx"], par["nt"]),
    )

    Pop = Patch3D(
        Op,
        dims=dims,  # (
        #    par["ny"] * par["winsy"],
        #    par["nx"] * par["winsx"],
        #    par["nt"] * par["winst"],
        # ),
        dimsd=(par["npy"], par["npx"], par["npt"]),
        nwin=(par["nwiny"], par["nwinx"], par["nwint"]),
        nover=(par["novery"], par["noverx"], par["novert"]),
        nop=(par["ny"], par["nx"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["ny"] * par["nx"] * par["nt"] * nwins[0] * nwins[1] * nwins[2],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones(
        (nwins[0], nwins[1], nwins[2], par["ny"], par["nx"], par["nt"]),
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch3D_singlepatch1(par, dtype):
    """Dot-test and inverse for Patch3D operator with single patch in fist dimension"""
    Op = MatrixMult(
        np.ones(
            (
                par["npy"] * par["nwinx"] * par["nwint"],
                par["npy"] * par["nx"] * par["nt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch3d_design(
        (par["npy"], par["npx"], par["npt"]),
        (par["npy"], par["nwinx"], par["nwint"]),
        (0, par["noverx"], par["novert"]),
        (par["npy"], par["nx"], par["nt"]),
    )

    Pop = Patch3D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npx"], par["npt"]),
        nwin=(par["npy"], par["nwinx"], par["nwint"]),
        nover=(0, par["noverx"], par["novert"]),
        nop=(par["npy"], par["nx"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert nwins[0] == 1
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["npy"] * par["nx"] * par["nt"] * nwins[0] * nwins[1] * nwins[2],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones(
        (nwins[0], nwins[1], nwins[2], par["npy"], par["nx"], par["nt"]),
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch3D_singlepatch2(par, dtype):
    """Dot-test and inverse for Patch3D operator with single patch in second dimension"""
    Op = MatrixMult(
        np.ones(
            (
                par["nwiny"] * par["npx"] * par["nwint"],
                par["ny"] * par["npx"] * par["nt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch3d_design(
        (par["npy"], par["npx"], par["npt"]),
        (par["nwiny"], par["npx"], par["nwint"]),
        (par["novery"], 0, par["novert"]),
        (par["ny"], par["npx"], par["nt"]),
    )

    Pop = Patch3D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npx"], par["npt"]),
        nwin=(par["nwiny"], par["npx"], par["nwint"]),
        nover=(par["novery"], 0, par["novert"]),
        nop=(par["ny"], par["npx"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert nwins[1] == 1
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["ny"] * par["npx"] * par["nt"] * nwins[0] * nwins[1] * nwins[2],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones(
        (nwins[0], nwins[1], nwins[2], par["ny"], par["npx"], par["nt"]),
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch3D_singlepatch12(par, dtype):
    """Dot-test and inverse for Patch3D operator with single patch in
    fist and second dimensions"""
    Op = MatrixMult(
        np.ones(
            (
                par["npy"] * par["npx"] * par["nwint"],
                par["npy"] * par["npx"] * par["nt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch3d_design(
        (par["npy"], par["npx"], par["npt"]),
        (par["npy"], par["npx"], par["nwint"]),
        (0, 0, par["novert"]),
        (par["npy"], par["npx"], par["nt"]),
    )

    Pop = Patch3D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npx"], par["npt"]),
        nwin=(par["npy"], par["npx"], par["nwint"]),
        nover=(0, 0, par["novert"]),
        nop=(par["npy"], par["npx"], par["nt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert nwins[0] == 1
    assert nwins[1] == 1
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["npy"] * par["npx"] * par["nt"] * nwins[0] * nwins[1] * nwins[2],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones(
        (nwins[0], nwins[1], nwins[2], par["npy"], par["npx"], par["nt"]),
        dtype=dtype,
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Patch3D_singlepatch3(par, dtype):
    """Dot-test and inverse for Patch3D operator with single patch in third dimension"""
    Op = MatrixMult(
        np.ones(
            (
                par["nwiny"] * par["nwinx"] * par["npt"],
                par["ny"] * par["nx"] * par["npt"],
            ),
            dtype=dtype,
        ),
        dtype=dtype,
    )

    nwins, dims, _, _ = patch3d_design(
        (par["npy"], par["npx"], par["npt"]),
        (par["nwiny"], par["nwinx"], par["npt"]),
        (par["novery"], par["noverx"], 0),
        (par["ny"], par["nx"], par["npt"]),
    )

    Pop = Patch3D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["npx"], par["npt"]),
        nwin=(par["nwiny"], par["nwinx"], par["npt"]),
        nover=(par["novery"], par["noverx"], 0),
        nop=(par["ny"], par["nx"], par["npt"]),
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["ny"] * par["nx"] * par["npt"] * nwins[0] * nwins[1] * nwins[2],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )
    x = np.ones(
        (nwins[0], nwins[1], nwins[2], par["ny"], par["nx"], par["npt"]),
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y
    xinv = cgls(Pop, y, niter=50)[0]

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_almost_equal(x, xinv, decimal=3 if dtype == np.float32 else 8)
