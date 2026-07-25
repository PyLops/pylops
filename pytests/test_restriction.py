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

from pylops.basicoperators import Restriction
from pylops.utils import dottest

par1 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "inplace": "True",
}  # real (fp64), inplace
par1s = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float32",
    "inplace": "True",
}  # real (fp32), inplace
par1j = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "inplace": "True",
}  # complex, inplace
par2 = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float64",
    "inplace": "False",
}  # real (fp64), out of place
par2s = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 0,
    "dtype": "float32",
    "inplace": "False",
}  # real (fp32), out of place
par2j = {
    "ny": 21,
    "nx": 11,
    "nt": 20,
    "imag": 1j,
    "dtype": "complex128",
    "inplace": "False",
}  # complex, out of place

# subsampling factor
perc_subsampling = 0.4


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par2), (par2s), (par2j)])
def test_Restriction_1dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 1d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(par["nx"], iava, inplace=par["inplace"], dtype=par["dtype"])
    assert dottest(
        Rop,
        Nsub,
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones(par["nx"], dtype=dtype) + par["imag"] * np.ones(par["nx"], dtype=dtype)
    y = Rop * x
    x1 = Rop.H * y
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par2), (par2s), (par2j)])
def test_Restriction_1dsignal_repeated(par):
    """Dot-test, forward and adjoint for Restriction operator for 1d signal
    with repeated incides (adjoint must sum over them)"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    # Force 2 repetitions
    iava[1] = iava[0]
    iava[-1] = iava[-2]

    Rop = Restriction(par["nx"], iava, inplace=par["inplace"], dtype=par["dtype"])
    assert dottest(
        Rop,
        Nsub,
        par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.arange(par["nx"], dtype=dtype) + par["imag"] * np.arange(
        par["nx"], dtype=dtype
    )
    y = Rop * x
    x1 = Rop.H * y

    x1ana_iava = np.zeros_like(x)
    x1ana_iava[iava[1:-1]] = x[iava[1:-1]]
    x1ana_iava[iava[0]] += x[iava[0]]
    x1ana_iava[iava[-1]] += x[iava[-1]]

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(x1, x1ana_iava)


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par2), (par2s), (par2j)])
def test_Restriction_2dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 2d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = np.ones((par["nx"], par["nt"]), dtype=dtype) + par["imag"] * np.ones(
        (par["nx"], par["nt"]), dtype=dtype
    )

    # 1st direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(
        (par["nx"], par["nt"]), iava, axis=0, inplace=par["inplace"], dtype=par["dtype"]
    )
    assert dottest(
        Rop,
        Nsub * par["nt"],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Rop * x.ravel()).reshape(Nsub, par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["nx"], par["nt"])
    y1_fromflat = np.asarray(Rop.mask(x.ravel()))
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, y1_fromflat.reshape(par["nx"], par["nt"])[iava])
    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])

    # 2nd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    Rop = Restriction(
        (par["nx"], par["nt"]), iava, axis=1, inplace=par["inplace"], dtype=par["dtype"]
    )
    assert dottest(
        Rop,
        par["nx"] * Nsub,
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Rop * x.ravel()).reshape(par["nx"], Nsub)
    x1 = (Rop.H * y.ravel()).reshape(par["nx"], par["nt"])
    y1_fromflat = np.asarray(Rop.mask(x.ravel()))
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, y1_fromflat[:, iava])
    assert_array_almost_equal(y, y1[:, iava])
    assert_array_almost_equal(x[:, iava], x1[:, iava])


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par2), (par2s), (par2j)])
def test_Restriction_3dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 3d signal"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    x = np.ones((par["ny"], par["nx"], par["nt"]), dtype=dtype) + par["imag"] * np.ones(
        (par["ny"], par["nx"], par["nt"]), dtype=dtype
    )

    # 1st direction
    Nsub = int(np.round(par["ny"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["ny"]))[:Nsub])

    Rop = Restriction(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=0,
        inplace=par["inplace"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        Nsub * par["nx"] * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Rop * x.ravel()).reshape(Nsub, par["nx"], par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = np.asarray(Rop.mask(x.ravel()))
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(
        y, y1_fromflat.reshape(par["ny"], par["nx"], par["nt"])[iava]
    )
    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])

    # 2nd direction
    Nsub = int(np.round(par["nx"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

    Rop = Restriction(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=1,
        inplace=par["inplace"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        par["ny"] * Nsub * par["nt"],
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Rop * x.ravel()).reshape(par["ny"], Nsub, par["nt"])
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = np.asarray(Rop.mask(x.ravel()))
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, y1_fromflat[:, iava])
    assert_array_almost_equal(y, y1[:, iava])
    assert_array_almost_equal(x[:, iava], x1[:, iava])

    # 3rd direction
    Nsub = int(np.round(par["nt"] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par["nt"]))[:Nsub])

    Rop = Restriction(
        (par["ny"], par["nx"], par["nt"]),
        iava,
        axis=2,
        inplace=par["inplace"],
        dtype=par["dtype"],
    )
    assert dottest(
        Rop,
        par["ny"] * par["nx"] * Nsub,
        par["ny"] * par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = (Rop * x.ravel()).reshape(par["ny"], par["nx"], Nsub)
    x1 = (Rop.H * y.ravel()).reshape(par["ny"], par["nx"], par["nt"])
    y1_fromflat = np.asarray(Rop.mask(x.ravel()))
    y1 = np.asarray(Rop.mask(x))

    assert y.dtype == par["dtype"]
    assert x1.dtype == par["dtype"]
    assert_array_almost_equal(y, y1_fromflat[:, :, iava])
    assert_array_almost_equal(y, y1[:, :, iava])
    assert_array_almost_equal(x[:, :, iava], x1[:, :, iava])
