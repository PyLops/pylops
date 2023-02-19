import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import Shift
from pylops.utils import dottest
from pylops.utils.wavelets import gaussian

par1 = {"nt": 41, "nx": 41, "ny": 11, "imag": 0, "dtype": "float64"}  # square real
par2 = {
    "nt": 41,
    "nx": 21,
    "ny": 11,
    "imag": 0,
    "dtype": "float64",
}  # overdetermined real
par1j = {
    "nt": 41,
    "nx": 41,
    "ny": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # square complex
par2j = {
    "nt": 41,
    "nx": 21,
    "ny": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # overdetermined complex


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_Shift1D(par):
    """Dot-test and inversion for Shift operator on 1d data"""
    np.random.seed(0)
    shift = 5.5
    x = (
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
    )

    Sop = Shift(
        par["nt"], shift, real=True if par["imag"] == 0 else False, dtype=par["dtype"]
    )
    assert dottest(Sop, par["nt"], par["nt"], complexflag=0 if par["imag"] == 0 else 3)

    xlsqr = lsqr(Sop, Sop * x, damp=1e-20, iter_lim=200, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Shift2D(par):
    """Dot-test and inversion for Shift operator on 2d data"""
    np.random.seed(0)
    shift = 5.5

    # 1st axis
    x = (
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
    )
    x = np.outer(x, np.ones(par["nx"]))
    Sop = Shift(
        (par["nt"], par["nx"]),
        shift,
        axis=0,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
    )
    xlsqr = lsqr(Sop, Sop * x.ravel(), damp=1e-20, iter_lim=200, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xlsqr, decimal=1)

    # 2nd axis
    x = (
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
    )
    x = np.outer(x, np.ones(par["nx"])).T
    Sop = Shift(
        (par["nx"], par["nt"]),
        shift,
        axis=1,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
    )
    xlsqr = lsqr(Sop, Sop * x.ravel(), damp=1e-20, iter_lim=200, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Shift2Dvariable(par):
    """Dot-test and inversion for Shift operator on 2d data with variable shift"""
    np.random.seed(0)
    shift = np.arange(par["nx"])

    # 1st axis
    x = (
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
    )
    x = np.outer(x, np.ones(par["nx"]))
    Sop = Shift(
        (par["nt"], par["nx"]),
        shift,
        axis=0,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
    )
    xlsqr = lsqr(Sop, Sop * x.ravel(), damp=1e-20, iter_lim=200, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xlsqr, decimal=1)

    # 2nd axis
    x = (
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0]
    )
    x = np.outer(x, np.ones(par["nx"])).T
    Sop = Shift(
        (par["nx"], par["nt"]),
        shift,
        axis=1,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
    )
    xlsqr = lsqr(Sop, Sop * x.ravel(), damp=1e-20, iter_lim=200, atol=1e-8, btol=1e-8, show=0)[0]
    assert_array_almost_equal(x.ravel(), xlsqr, decimal=1)
