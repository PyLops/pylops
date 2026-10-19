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

# Cover both precisions and copy modes for negative/positive index aliases.
par3 = dict(par1, inplace=True)
par3s = dict(par1s, inplace=True)
par3j = dict(par1j, inplace=True)
par3js = dict(par1j, dtype="complex64", inplace=True)
par4 = dict(par2, inplace=False)
par4s = dict(par2s, inplace=False)
par4j = dict(par2j, inplace=False)
par4js = dict(par2j, dtype="complex64", inplace=False)

np.random.seed(10)


# subsampling factor
perc_subsampling = 0.4


@pytest.mark.parametrize(
    "par", [par3, par3s, par3j, par3js, par4, par4s, par4j, par4js]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_Restriction_mask_repeated_aliases(par, axis):
    """Reject masks with repeated locations expressed using different indices."""
    dims = (par["nx"], par["nt"])
    iava = np.array([dims[axis] - 1, -1])
    Rop = Restriction(dims, iava, axis=axis, dtype=par["dtype"])
    x = np.random.normal(size=dims) + par["imag"] * np.random.normal(size=dims)
    with pytest.raises(ValueError, match="index in iava is repeated"):
        Rop.mask(x.astype(par["dtype"]))


@pytest.mark.parametrize(
    "par", [par3, par3s, par3j, par3js, par4, par4s, par4j, par4js]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_Restriction_repeated_aliases(par, axis):
    """Sum every sample at locations shared by negative and positive indices."""
    dims = (par["nx"], par["nt"])
    iava = np.array([0, -dims[axis], dims[axis] - 1, -1])
    Rop = Restriction(dims, iava, axis=axis, inplace=par["inplace"], dtype=par["dtype"])
    rdtype = np.empty(0, dtype=par["dtype"]).real.dtype
    assert dottest(
        Rop,
        len(iava) * dims[1 - axis % len(dims)],
        par["nx"] * par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if rdtype == np.float32 else 1e-6,
        backend=backend,
    )
    assert Rop.repeated
    assert_array_almost_equal(Rop.iava, iava)

    x = np.random.normal(size=dims) + par["imag"] * np.random.normal(size=dims)
    x = x.astype(par["dtype"])
    y = np.random.normal(size=Rop.dimsd) + par["imag"] * np.random.normal(
        size=Rop.dimsd
    )
    y = y.astype(par["dtype"])
    x_before, y_before = x.copy(), y.copy()
    expected = np.zeros_like(x)
    # Add one observation at a time, without fancy-indexed assignment.
    expected_axis = np.moveaxis(expected, axis, 0)
    y_axis = np.moveaxis(y, axis, 0)
    for row, index in enumerate(iava):
        expected_axis[index] += y_axis[row]

    forward = Rop.matvec(x.ravel()).reshape(Rop.dimsd)
    adjoint = Rop.rmatvec(y.ravel()).reshape(dims)
    assert_array_almost_equal(forward, np.take(x, iava, axis=axis))
    assert_array_almost_equal(adjoint, expected)
    assert forward.dtype == par["dtype"]
    assert adjoint.dtype == par["dtype"]
    assert_array_almost_equal(x, x_before)
    assert_array_almost_equal(y, y_before)


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
