import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.signalprocessing import Bilinear, Interp
from pylops.utils import dottest

par1 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "kind": "nearest",
}  # real, nearest
par2 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "kind": "nearest",
}  # complex, nearest
par3 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "kind": "linear",
}  # real, linear
par4 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "kind": "linear",
}  # complex, linear
par5 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "kind": "sinc",
}  # real, sinc
par6 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "kind": "sinc",
}  # complex, sinc

# subsampling factor
perc_subsampling = 0.4


def test_sincinterp():
    """Check accuracy of sinc interpolation of subsampled version of input
    signal
    """
    nt = 81
    dt = 0.004
    t = np.arange(nt) * dt

    ntsub = 10
    dtsub = dt / ntsub
    tsub = np.arange(nt * ntsub) * dtsub
    tsub = tsub[: np.where(tsub == t[-1])[0][0] + 1]

    x = (
        np.sin(2 * np.pi * 10 * t)
        + 0.4 * np.sin(2 * np.pi * 20 * t)
        - 2 * np.sin(2 * np.pi * 5 * t)
    )
    xsub = (
        np.sin(2 * np.pi * 10 * tsub)
        + 0.4 * np.sin(2 * np.pi * 20 * tsub)
        - 2 * np.sin(2 * np.pi * 5 * tsub)
    )

    iava = tsub[20:-20] / (dtsub * ntsub)  # exclude edges
    SI1op, iava = Interp(nt, iava, kind="sinc", dtype="float64")
    y = SI1op * x
    assert_array_almost_equal(xsub[20:-20], y, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Interp_1dsignal(par):
    """Dot-test and forward for Interp operator for 1d signal"""
    np.random.seed(1)
    x = np.random.normal(0, 1, par["nx"]) + par["imag"] * np.random.normal(
        0, 1, par["nx"]
    )

    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par["nx"], iava, kind=par["kind"], dtype=par["dtype"])
    assert dottest(Iop, Nsub, par["nx"], complexflag=0 if par["imag"] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par["nx"], iava + 0.3, kind=par["kind"], dtype=par["dtype"])
    assert dottest(Iop, Nsub, par["nx"], complexflag=0 if par["imag"] == 0 else 3)

    # repeated indeces
    with pytest.raises(ValueError):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(par["nx"], iava_rep + 0.3, kind=par["kind"], dtype=par["dtype"])

    # forward
    y = Iop * x
    ydec = Idecop * x

    assert_array_almost_equal(y, x[iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[iava])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Interp_2dsignal(par):
    """Dot-test and forward for Restriction operator for 2d signal"""
    np.random.seed(1)
    x = np.random.normal(0, 1, (par["nx"], par["nt"])) + par["imag"] * np.random.normal(
        0, 1, (par["nx"], par["nt"])
    )

    # 1st direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(
        (par["nx"], par["nt"]),
        iava,
        axis=0,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Iop,
        Nsub * par["nt"],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Idecop, _ = Interp(
        (par["nx"], par["nt"]),
        iava + 0.3,
        axis=0,
        kind=par["kind"],
        dtype=par["dtype"],
    )

    # repeated indeces
    with pytest.raises(ValueError):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(
            (par["nx"], par["nt"]),
            iava_rep + 0.3,
            axis=0,
            kind=par["kind"],
            dtype=par["dtype"],
        )

    y = (Iop * x.ravel()).reshape(Nsub, par["nt"])
    ydec = (Idecop * x.ravel()).reshape(Nsub, par["nt"])

    assert_array_almost_equal(y, x[iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[iava])

    # 2nd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(
        (par["nx"], par["nt"]),
        iava,
        axis=1,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Iop,
        par["nx"] * Nsub,
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Idecop, _ = Interp(
        (par["nx"], par["nt"]),
        iava + 0.3,
        axis=1,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Idecop,
        par["nx"] * Nsub,
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Iop * x.ravel()).reshape(par["nx"], Nsub)
    ydec = (Idecop * x.ravel()).reshape(par["nx"], Nsub)

    assert_array_almost_equal(y, x[:, iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[:, iava])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Interp_3dsignal(par):
    """Dot-test and forward  for Interp operator for 3d signal"""
    np.random.seed(1)
    x = np.random.normal(0, 1, (par["ny"], par["nx"], par["nt"])) + par[
        "imag"
    ] * np.random.normal(0, 1, (par["ny"], par["nx"], par["nt"]))

    # 1st direction
    Nsub = int(np.round(par["ny"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["ny"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=0,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Iop,
        Nsub * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Idecop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava + 0.3,
        axis=0,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Idecop,
        Nsub * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # repeated indeces
    with pytest.raises(ValueError):
        iava_rep = iava.copy()
        iava_rep[-2] = 0
        iava_rep[-1] = 0
        _, _ = Interp(
            (par["ny"], par["nx"], par["nt"]),
            iava_rep + 0.3,
            axis=0,
            kind=par["kind"],
            dtype=par["dtype"],
        )

    y = (Iop * x.ravel()).reshape(Nsub, par["nx"], par["nt"])
    ydec = (Idecop * x.ravel()).reshape(Nsub, par["nx"], par["nt"])

    assert_array_almost_equal(y, x[iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[iava])

    # 2nd direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=1,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Iop,
        par["ny"] * Nsub * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Idecop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava + 0.3,
        axis=1,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Idecop,
        par["ny"] * Nsub * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Iop * x.ravel()).reshape(par["ny"], Nsub, par["nt"])
    ydec = (Idecop * x.ravel()).reshape(par["ny"], Nsub, par["nt"])

    assert_array_almost_equal(y, x[:, iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[:, iava])

    # 3rd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=2,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Iop,
        par["ny"] * par["nx"] * Nsub,
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Idecop, _ = Interp(
        (par["ny"], par["nx"], par["nt"]),
        iava + 0.3,
        axis=2,
        kind=par["kind"],
        dtype=par["dtype"],
    )
    assert dottest(
        Idecop,
        par["ny"] * par["nx"] * Nsub,
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    y = (Iop * x.ravel()).reshape(par["ny"], par["nx"], Nsub)
    ydec = (Idecop * x.ravel()).reshape(par["ny"], par["nx"], Nsub)

    assert_array_almost_equal(y, x[:, :, iava])
    if par["kind"] == "nearest":
        assert_array_almost_equal(ydec, x[:, :, iava])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Bilinear_2dsignal(par):
    """Dot-test and forward for Interp operator for 2d signal"""
    np.random.seed(1)
    x = np.random.normal(0, 1, (par["nx"], par["nt"])) + par["imag"] * np.random.normal(
        0, 1, (par["nx"], par["nt"])
    )

    # fixed indeces
    iava = np.vstack((np.arange(0, 10), np.arange(0, 10)))
    Iop = Bilinear(iava, dims=(par["nx"], par["nt"]), dtype=par["dtype"])
    assert dottest(
        Iop, 10, par["nx"] * par["nt"], complexflag=0 if par["imag"] == 0 else 3
    )

    # decimal indeces
    Nsub = int(np.round(par["nx"] * par["nt"] * perc_subsampling))
    iavadec = np.vstack(
        (
            np.random.uniform(0, par["nx"] - 1, Nsub),
            np.random.uniform(0, par["nt"] - 1, Nsub),
        )
    )
    Idecop = Bilinear(iavadec, dims=(par["nx"], par["nt"]), dtype=par["dtype"])
    assert dottest(
        Idecop, Nsub, par["nx"] * par["nt"], complexflag=0 if par["imag"] == 0 else 3
    )

    # repeated indeces
    with pytest.raises(ValueError):
        iava_rep = iava.copy()
        iava_rep[-2] = [0, 0]
        iava_rep[-1] = [0, 0]
        _, _ = Bilinear(iava_rep, dims=(par["nx"], par["nt"]), dtype=par["dtype"])

    y = Iop * x.ravel()
    assert_array_almost_equal(y, x[iava[0], iava[1]])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Bilinear_2dsignal_flatten(par):
    """Dot-test and forward for Interp operator for 2d signal with forceflat"""
    np.random.seed(1)
    dims = (par["nx"], par["nt"])
    flat_dims = par["nx"] * par["nt"]
    dimsd = 10

    x = np.random.normal(0, 1, dims) + par["imag"] * np.random.normal(0, 1, dims)

    iava = np.vstack((np.arange(0, dimsd), np.arange(0, dimsd)))
    Iop_True = Bilinear(iava, dims=dims, dtype=par["dtype"], forceflat=True)
    y = Iop_True @ x
    xadj = Iop_True.H @ y
    assert y.shape == (dimsd,)
    assert xadj.shape == (flat_dims,)

    Iop_None = Bilinear(iava, dims=dims, dtype=par["dtype"])
    y = Iop_None @ x
    xadj = Iop_None.H @ y
    assert y.shape == (dimsd,)
    assert xadj.shape == dims


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Bilinear_3dsignal(par):
    """Dot-test and forward for Interp operator for 3d signal"""
    np.random.seed(1)
    x = np.random.normal(0, 1, (par["ny"], par["nx"], par["nt"])) + par[
        "imag"
    ] * np.random.normal(0, 1, (par["ny"], par["nx"], par["nt"]))

    # fixed indeces
    iava = np.vstack((np.arange(0, 10), np.arange(0, 10)))
    Iop = Bilinear(iava, dims=(par["ny"], par["nx"], par["nt"]), dtype=par["dtype"])
    assert dottest(
        Iop,
        10 * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # decimal indeces
    Nsub = int(np.round(par["ny"] * par["nt"] * perc_subsampling))
    iavadec = np.vstack(
        (
            np.random.uniform(0, par["ny"] - 1, Nsub),
            np.random.uniform(0, par["nx"] - 1, Nsub),
        )
    )
    Idecop = Bilinear(
        iavadec, dims=(par["ny"], par["nx"], par["nt"]), dtype=par["dtype"]
    )
    assert dottest(
        Idecop,
        Nsub * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
    )

    # repeated indeces
    with pytest.raises(ValueError):
        iava_rep = iava.copy()
        iava_rep[-2] = [0, 0]
        iava_rep[-1] = [0, 0]
        _, _ = Bilinear(
            iava_rep, dims=(par["ny"], par["nx"], par["nt"]), dtype=par["dtype"]
        )

    y = Iop * x.ravel()
    assert_array_almost_equal(y, x[iava[0], iava[1]].ravel())
