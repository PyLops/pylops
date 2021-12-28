import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import Restriction
from pylops.utils import dottest

par1 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "inplace": "True",
}  # real, inplace
par2 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "inplace": "True",
}  # complex, inplace
par3 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "inplace": "False",
}  # real, out of place
par4 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "inplace": "False",
}  # complex, out of place

# subsampling factor
perc_subsampling = 0.4


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Restriction_1dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 1d signal"""
    np.random.seed(10)

    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(par["nx"], iava, inplace=par["dtype"], dtype=par["dtype"])
    assert dottest(Rop, Nsub, par["nx"], complexflag=0 if par["imag"] == 0 else 3)

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Rop * x
    x1 = Rop.H * y
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Restriction_2dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 2d signal"""
    np.random.seed(10)

    x = np.ones((par["nx"], par["nt"])) + par["imag"] * np.ones((par["nx"], par["nt"]))

    # 1st direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(
        par["nx"] * par["nt"],
        iava,
        dims=(par["nx"], par["nt"]),
        dir=0,
        inplace=par["dtype"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        Nsub * par["nt"],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Rop * x.ravel()).reshape(Nsub, par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["nx"], par["nt"])
    y1_fromflat = Rop.mask(x.ravel())
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1_fromflat.reshape(par["nx"], par["nt"])[iava])
    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])

    # 2nd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    Rop = Restriction(
        par["nx"] * par["nt"],
        iava,
        dims=(par["nx"], par["nt"]),
        dir=1,
        inplace=par["dtype"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        par["nx"] * Nsub,
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Rop * x.ravel()).reshape(par["nx"], Nsub)
    x1 = (Rop.H * y.ravel()).reshape(par["nx"], par["nt"])
    y1_fromflat = Rop.mask(x.ravel())
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1_fromflat[:, iava])
    assert_array_almost_equal(y, y1[:, iava])
    assert_array_almost_equal(x[:, iava], x1[:, iava])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Restriction_3dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 3d signal"""
    np.random.seed(10)

    x = np.ones((par["ny"], par["nx"], par["nt"])) + par["imag"] * np.ones(
        (par["ny"], par["nx"], par["nt"])
    )

    # 1st direction
    Nsub = int(np.round(par["ny"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["ny"]))[:Nsub])

    Rop = Restriction(
        par["ny"] * par["nx"] * par["nt"],
        iava,
        dims=(par["ny"], par["nx"], par["nt"]),
        dir=0,
        inplace=par["dtype"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        Nsub * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Rop * x.ravel()).reshape(Nsub, par["nx"], par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = Rop.mask(x.ravel())
    y1 = Rop.mask(x)

    assert_array_almost_equal(
        y, y1_fromflat.reshape(par["ny"], par["nx"], par["nt"])[iava]
    )
    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])

    # 2nd direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(
        par["ny"] * par["nx"] * par["nt"],
        iava,
        dims=(par["ny"], par["nx"], par["nt"]),
        dir=1,
        inplace=par["dtype"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        par["ny"] * Nsub * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Rop * x.ravel()).reshape(par["ny"], Nsub, par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = Rop.mask(x.ravel())
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1_fromflat[:, iava])
    assert_array_almost_equal(y, y1[:, iava])
    assert_array_almost_equal(x[:, iava], x1[:, iava])

    # 3rd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    Rop = Restriction(
        par["ny"] * par["nx"] * par["nt"],
        iava,
        dims=(par["ny"], par["nx"], par["nt"]),
        dir=2,
        inplace=par["dtype"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        par["ny"] * par["nx"] * Nsub,
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Rop * x.ravel()).reshape(par["ny"], par["nx"], Nsub)
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = Rop.mask(x.ravel())
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1_fromflat[:, :, iava])
    assert_array_almost_equal(y, y1[:, :, iava])
    assert_array_almost_equal(x[:, :, iava], x1[:, :, iava])
