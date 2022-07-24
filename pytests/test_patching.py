import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.signalprocessing import Patch2D, Patch3D
from pylops.signalprocessing.Patch2D import Patch2Ddesign
from pylops.signalprocessing.Patch3D import Patch3Ddesign
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
    "tapertype": "hanning",
}  # overlap, with taper


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Patch2D(par):
    """Dot-test and inverse for Patch2D operator"""
    Op = MatrixMult(np.ones((par["nwiny"] * par["nwint"], par["ny"] * par["nt"])))

    nwins, dims, mwin_inends, dwin_inends = Patch2Ddesign(
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
    )
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["ny"] * par["nt"] * nwins[0] * nwins[1],
    )
    x = np.ones((par["ny"] * nwins[0], par["nt"] * nwins[1]))
    y = Pop * x.ravel()

    xinv = LinearOperator(Pop) / y
    assert_array_almost_equal(x.ravel(), xinv)


@pytest.mark.parametrize("par", [(par1), (par4)])
def test_Patch2D_scalings(par):
    """Dot-test and inverse for Patch2D operator with scalings"""
    Op = MatrixMult(np.ones((par["nwiny"] * par["nwint"], par["ny"] * par["nt"])))
    scalings = np.arange(par["nwiny"] * par["nwint"]) + 1.0

    nwins, dims, mwin_inends, dwin_inends = Patch2Ddesign(
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
        scalings=scalings,
    )
    assert dottest(
        Pop,
        par["npy"] * par["npt"],
        par["ny"] * par["nt"] * nwins[0] * nwins[1],
    )
    x = np.ones((par["ny"] * nwins[0], par["nt"] * nwins[1]))
    y = Pop * x.ravel()

    xinv = LinearOperator(Pop) / y
    assert_array_almost_equal(x.ravel(), xinv)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Patch3D(par):
    """Dot-test and inverse for Patch3D operator"""
    Op = MatrixMult(
        np.ones(
            (
                par["nwiny"] * par["nwinx"] * par["nwint"],
                par["ny"] * par["nx"] * par["nt"],
            )
        )
    )

    nwins, dims, mwin_inends, dwin_inends = Patch3Ddesign(
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
    )
    assert dottest(
        Pop,
        par["npy"] * par["npx"] * par["npt"],
        par["ny"] * par["nx"] * par["nt"] * nwins[0] * nwins[1] * nwins[2],
    )
    x = np.ones((par["ny"] * nwins[0], par["nx"] * nwins[1], par["nt"] * nwins[2]))
    y = Pop * x.ravel()

    xinv = LinearOperator(Pop) / y
    assert_array_almost_equal(x.ravel(), xinv)
