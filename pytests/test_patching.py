import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.signalprocessing import Patch2D
from pylops.utils import dottest

par1 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 5,
    "novery": 0,
    "winsy": 3,
    "npt": 10,
    "nwint": 5,
    "novert": 0,
    "winst": 2,
    "tapertype": None,
}  # no overlap, no taper
par2 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 5,
    "novery": 0,
    "winsy": 3,
    "npt": 10,
    "nwint": 5,
    "novert": 0,
    "winst": 2,
    "tapertype": "hanning",
}  # no overlap, with taper
par3 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    "winsy": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    "winst": 4,
    "tapertype": None,
}  # overlap, no taper
par4 = {
    "ny": 6,
    "nx": 7,
    "nt": 10,
    "npy": 15,
    "nwiny": 7,
    "novery": 3,
    "winsy": 3,
    "npt": 10,
    "nwint": 4,
    "novert": 2,
    "winst": 4,
    "tapertype": "hanning",
}  # overlap, with taper


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Patch2D(par):
    """Dot-test and inverse for Patch2D operator"""
    Op = MatrixMult(np.ones((par["nwiny"] * par["nwint"], par["ny"] * par["nt"])))

    Pop = Patch2D(
        Op,
        dims=(par["ny"] * par["winsy"], par["nt"] * par["winst"]),
        dimsd=(par["npy"], par["npt"]),
        nwin=(par["nwiny"], par["nwint"]),
        nover=(par["novery"], par["novert"]),
        nop=(par["ny"], par["nt"]),
        tapertype=par["tapertype"],
    )
    assert dottest(
        Pop, par["npy"] * par["nt"], par["ny"] * par["nt"] * par["winsy"] * par["winst"]
    )
    x = np.ones((par["ny"] * par["winsy"], par["nt"] * par["winst"]))
    y = Pop * x.ravel()

    xinv = LinearOperator(Pop) / y
    assert_array_almost_equal(x.ravel(), xinv)
