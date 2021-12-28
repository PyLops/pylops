import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy.sparse.linalg import lsqr

from pylops.basicoperators import Transpose
from pylops.utils import dottest

par1 = {"ny": 21, "nx": 11, "nt": 20, "imag": 0, "dtype": "float32"}  # real
par2 = {"ny": 21, "nx": 11, "nt": 20, "imag": 1j, "dtype": "complex64"}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Transpose_2dsignal(par):
    """Dot-test and adjoint for Transpose operator for 2d signals"""
    dims = (par["ny"], par["nx"])
    x = np.arange(par["ny"] * par["nx"]).reshape(dims) + par["imag"] * np.arange(
        par["ny"] * par["nx"]
    ).reshape(dims)

    Top = Transpose(dims=dims, axes=(1, 0), dtype=par["dtype"])
    assert dottest(
        Top, np.prod(dims), np.prod(dims), complexflag=0 if par["imag"] == 0 else 3
    )
    y = Top * x.ravel()
    xadj = Top.H * y

    y = y.reshape(Top.dimsd)
    xadj = xadj.reshape(Top.dims)

    assert_equal(x, xadj)
    assert_equal(y, x.T)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Transpose_3dsignal(par):
    """Dot-test and adjoint for Transpose operator for 3d signals"""
    dims = (par["ny"], par["nx"], par["nt"])
    x = np.arange(par["ny"] * par["nx"] * par["nt"]).reshape(dims) + par[
        "imag"
    ] * np.arange(par["ny"] * par["nx"] * par["nt"]).reshape(dims)

    Top = Transpose(dims=dims, axes=(2, 1, 0))
    assert dottest(
        Top, np.prod(dims), np.prod(dims), complexflag=0 if par["imag"] == 0 else 3
    )

    y = Top * x.ravel()
    xadj = Top.H * y

    y = y.reshape(Top.dimsd)
    xadj = xadj.reshape(Top.dims)

    assert_equal(x, xadj)
