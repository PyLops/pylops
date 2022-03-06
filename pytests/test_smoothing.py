import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse.linalg import lsqr

from pylops.basicoperators import Smoothing1D, Smoothing2D
from pylops.utils import dottest

par1 = {"nz": 10, "ny": 30, "nx": 20, "axis": 0}  # even, first direction
par2 = {"nz": 11, "ny": 51, "nx": 31, "axis": 0}  # odd, first direction
par3 = {"nz": 10, "ny": 30, "nx": 20, "axis": 1}  # even, second direction
par4 = {"nz": 11, "ny": 51, "nx": 31, "axis": 1}  # odd, second direction
par5 = {"nz": 10, "ny": 30, "nx": 20, "axis": 2}  # even, third direction
par6 = {"nz": 11, "ny": 51, "nx": 31, "axis": 2}  # odd, third direction

np.random.seed(0)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Smoothing1D(par):
    """Dot-test and inversion for smoothing"""
    # 1d kernel on 1d signal
    D1op = Smoothing1D(nsmooth=5, dims=par["nx"], dtype="float64")
    assert dottest(D1op, par["nx"], par["nx"])

    x = np.random.normal(0, 1, par["nx"])
    y = D1op * x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # 1d kernel on 2d signal
    D1op = Smoothing1D(
        nsmooth=5, dims=(par["ny"], par["nx"]), axis=par["axis"], dtype="float64"
    )
    assert dottest(D1op, par["ny"] * par["nx"], par["ny"] * par["nx"])

    x = np.random.normal(0, 1, (par["ny"], par["nx"])).ravel()
    y = D1op * x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # 1d kernel on 3d signal
    D1op = Smoothing1D(
        nsmooth=5,
        dims=(par["nz"], par["ny"], par["nx"]),
        axis=par["axis"],
        dtype="float64",
    )
    assert dottest(
        D1op,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-3,
    )

    x = np.random.normal(0, 1, (par["nz"], par["ny"], par["nx"])).ravel()
    y = D1op * x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Smoothing2D(par):
    """Dot-test for smoothing"""
    # 2d kernel on 2d signal
    if par["axis"] < 2:
        D2op = Smoothing2D(nsmooth=(5, 5), dims=(par["ny"], par["nx"]), dtype="float64")
        assert dottest(D2op, par["ny"] * par["nx"], par["ny"] * par["nx"], rtol=1e-3)

        # forward
        x = np.zeros((par["ny"], par["nx"]))
        x[par["ny"] // 2, par["nx"] // 2] = 1.0
        x = x.ravel()
        y = D2op * x
        y = y.reshape(par["ny"], par["nx"])
        assert_array_almost_equal(
            y[par["ny"] // 2 - 2 : par["ny"] // 2 + 3 :, par["nx"] // 2],
            np.ones(5) / 25,
        )
        assert_array_almost_equal(
            y[par["ny"] // 2, par["nx"] // 2 - 2 : par["nx"] // 2 + 3], np.ones(5) / 25
        )
        # inverse
        xlsqr = lsqr(D2op, y.ravel(), damp=1e-10, iter_lim=400, show=0)[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2d kernel on 3d signal
    axes = list(range(3))
    axes.remove(par["axis"])
    D2op = Smoothing2D(
        nsmooth=(5, 5),
        dims=(par["nz"], par["ny"], par["nx"]),
        axes=axes,
        dtype="float64",
    )
    assert dottest(
        D2op, par["nz"] * par["ny"] * par["nx"], par["nz"] * par["ny"] * par["nx"]
    )

    # forward
    x = np.zeros((par["nz"], par["ny"], par["nx"]))
    x[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2] = 1.0
    x = x.ravel()
    y = D2op * x
    y = y.reshape(par["nz"], par["ny"], par["nx"])

    if par["axis"] == 0:
        assert_array_almost_equal(
            y[par["nz"] // 2, par["ny"] // 2 - 2 : par["ny"] // 2 + 3, par["nx"] // 2],
            np.ones(5) / 25,
        )
        assert_array_almost_equal(
            y[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2 - 2 : par["nx"] // 2 + 3],
            np.ones(5) / 25,
        )
    elif par["axis"] == 1:
        assert_array_almost_equal(
            y[par["nz"] // 2 - 2 : par["nz"] // 2 + 3, par["ny"] // 2, par["nx"] // 2],
            np.ones(5) / 25,
        )
        assert_array_almost_equal(
            y[par["nz"] // 2, par["ny"] // 2, par["nx"] // 2 - 2 : par["nx"] // 2 + 3],
            np.ones(5) / 25,
        )
    elif par["axis"] == 2:
        assert_array_almost_equal(
            y[par["nz"] // 2 - 2 : par["nz"] // 2 + 3, par["ny"] // 2, par["nx"] // 2],
            np.ones(5) / 25,
        )
        assert_array_almost_equal(
            y[par["nz"] // 2, par["ny"] // 2 - 2 : par["ny"] // 2 + 3, par["nx"] // 2],
            np.ones(5) / 25,
        )

    # inverse
    xlsqr = lsqr(D2op, y.ravel(), damp=1e-10, iter_lim=400, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)
