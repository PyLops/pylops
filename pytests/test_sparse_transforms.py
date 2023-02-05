import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.signalprocessing import DCT
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}
par2 = {"ny": 11, "nx": 21, "imag": 0, "dtype": "float64"}
par3 = {"ny": 21, "nx": 21, "imag": 0, "dtype": "float64"}


@pytest.mark.parametrize("par", [(par1), (par3)])
def test_DCT1D(par):
    """Dot test for Discrete Cosine Transform Operator 1D"""

    t = np.arange(par["ny"])
    # testing for various types of dct

    for type in [1, 2, 3, 4]:
        Dct = DCT(dims=(par["ny"],), type=type, dtype=par["dtype"])

        assert dottest(Dct, par["ny"], par["ny"], rtol=1e-6, complexflag=0, verb=True)

        y = Dct.H * (Dct * t)

        assert_array_almost_equal(t, y, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_DCT2D(par):
    """Dot test for Discrete Cosine Transform Operator 2D"""

    t = np.outer(np.arange(par["ny"]), np.arange(par["nx"]))

    for type in [1, 2, 3, 4]:
        for axes in [0, 1]:
            Dct = DCT(dims=t.shape, type=type, axes=axes, dtype=par["dtype"])

            assert dottest(
                Dct,
                par["nx"] * par["ny"],
                par["nx"] * par["ny"],
                rtol=1e-6,
                complexflag=0,
                verb=True,
            )

            y = Dct.H * (Dct * t)

            assert_array_almost_equal(t, y, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_DCT3D(par):
    """Dot test for Discrete Cosine Transform Operator 3D"""

    t = np.random.rand(par["nx"], par["nx"], par["nx"])

    for type in [1, 2, 3, 4]:
        for axes in [0, 1, 2]:
            Dct = DCT(dims=t.shape, type=type, axes=axes, dtype=par["dtype"])

            assert dottest(
                Dct,
                par["nx"] * par["nx"] * par["nx"],
                par["nx"] * par["nx"] * par["nx"],
                rtol=1e-6,
                complexflag=0,
                verb=True,
            )

            y = Dct.H * (Dct * t)

            assert_array_almost_equal(t, y, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par3)])
def test_DCT_workers(par):
    """Dot test for Discrete Cosine Transform Operator with workers"""
    t = np.arange(par["ny"])

    Dct = DCT(dims=(par["ny"],), type=1, dtype=par["dtype"], workers=2)

    assert dottest(Dct, par["ny"], par["ny"], rtol=1e-6, complexflag=0, verb=True)

    y = Dct.H * (Dct * t)

    assert_array_almost_equal(t, y, decimal=3)
