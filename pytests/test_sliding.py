import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult
from pylops.signalprocessing import Sliding1D, Sliding2D, Sliding3D
from pylops.signalprocessing.sliding1d import sliding1d_design
from pylops.signalprocessing.sliding2d import sliding2d_design
from pylops.signalprocessing.sliding3d import sliding3d_design
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


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Sliding1D(par):
    """Dot-test and inverse for Sliding1D operator"""
    Op = MatrixMult(np.ones((par["nwiny"], par["ny"])))

    nwins, dim, mwin_inends, dwin_inends = sliding1d_design(
        par["npy"], par["nwiny"], par["novery"], par["ny"]
    )

    Slid = Sliding1D(
        Op,
        dim=dim,
        dimd=par["npy"],
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(Slid, par["npy"], par["ny"] * nwins)
    x = np.ones(par["ny"] * nwins)
    y = Slid * x.ravel()

    xinv = Slid / y
    assert_array_almost_equal(x.ravel(), xinv)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Sliding2D(par):
    """Dot-test and inverse for Sliding2D operator"""
    Op = MatrixMult(np.ones((par["nwiny"] * par["nt"], par["ny"] * par["nt"])))

    nwins, dims, mwin_inends, dwin_inends = sliding2d_design(
        (par["npy"], par["nt"]), par["nwiny"], par["novery"], (par["ny"], par["nt"])
    )
    Slid = Sliding2D(
        Op,
        dims=dims,
        dimsd=(par["npy"], par["nt"]),
        nwin=par["nwiny"],
        nover=par["novery"],
        tapertype=par["tapertype"],
        savetaper=par["savetaper"],
    )
    assert dottest(Slid, par["npy"] * par["nt"], par["ny"] * par["nt"] * nwins)
    x = np.ones((par["ny"] * nwins, par["nt"]))
    y = Slid * x.ravel()

    xinv = Slid / y
    assert_array_almost_equal(x.ravel(), xinv)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Sliding3D(par):
    """Dot-test and inverse for Sliding3D operator"""
    Op = MatrixMult(
        np.ones(
            (par["nwiny"] * par["nwinx"] * par["nt"], par["ny"] * par["nx"] * par["nt"])
        )
    )

    nwins, dims, mwin_inends, dwin_inends = sliding3d_design(
        (par["npy"], par["npx"], par["nt"]),
        (par["nwiny"], par["nwinx"]),
        (par["novery"], par["noverx"]),
        (par["ny"], par["nx"], par["nt"]),
    )

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
    )
    x = np.ones((par["ny"] * par["nx"] * nwins[0] * nwins[1], par["nt"]))
    y = Slid * x.ravel()

    xinv = Slid / y
    assert_array_almost_equal(x.ravel(), xinv)
