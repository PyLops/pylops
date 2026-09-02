import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"

import numpy as npp
import pytest
from scipy.ndimage import gaussian_filter

from pylops.optimization.basic import lsqr
from pylops.signalprocessing import Downsample2D
from pylops.signalprocessing.downsample2d import standard_deviation_from_attenuation
from pylops.utils import dottest

par1 = {
    "ny": 21,
    "nx": 15,
    "factors": 3,
    "imag": 0,
    "dtype": "float64",
}  # same factor, real
par2 = {
    "ny": 20,
    "nx": 16,
    "factors": (2, 4),
    "imag": 0,
    "dtype": "float64",
}  # different factors, real
par3 = {
    "ny": 11,
    "nx": 13,
    "factors": 1,
    "imag": 0,
    "dtype": "float64",
}  # unitary factor, real
par1j = {
    "ny": 21,
    "nx": 15,
    "factors": 3,
    "imag": 1j,
    "dtype": "complex128",
}  # same factor, complex
par2j = {
    "ny": 20,
    "nx": 16,
    "factors": (2, 4),
    "imag": 1j,
    "dtype": "complex128",
}  # different factors, complex


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": (10,)},
        {"dims": (10, 10), "axes": (0,)},
        {"dims": (10, 10), "factors": (2, 2, 2)},
        {"dims": (10, 10), "factors": 0},
        {"dims": (10, 10), "factors": 11},
        {"dims": (10, 10), "sigma": (1.0, 1.0, 1.0)},
        {"dims": (10, 10), "sigma": -1.0},
    ],
)
def test_Downsample2D_raises(kwargs):
    """Check input validation of Downsample2D"""
    with pytest.raises(ValueError):
        Downsample2D(**kwargs)


def test_Downsample2D_sigma():
    """Check that a null sigma leads to pure subsampling"""
    x = np.random.normal(0.0, 1.0, (12, 9))
    Dop = Downsample2D((12, 9), factors=(3, 3), sigma=0.0)
    assert Dop.h.shape == (1, 1)
    assert_array_almost_equal(Dop @ x, x[::3, ::3], decimal=10)


def test_Downsample2D_ndim():
    """Check that Downsample2D can be applied to a subset of axes of a
    3-dimensional array
    """
    Dop = Downsample2D((7, 9, 5), factors=2, axes=(0, 1))
    assert Dop.dimsd == (4, 5, 5)
    assert dottest(Dop, *Dop.shape, rtol=1e-6, backend=backend)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1j), (par2j)])
def test_Downsample2D(par):
    """Dot-test and shapes for Downsample2D"""
    Dop = Downsample2D(
        (par["ny"], par["nx"]), factors=par["factors"], dtype=par["dtype"]
    )
    factors = (
        (par["factors"], par["factors"])
        if isinstance(par["factors"], int)
        else par["factors"]
    )
    assert Dop.dimsd == (
        int(npp.ceil(par["ny"] / factors[0])),
        int(npp.ceil(par["nx"] / factors[1])),
    )
    assert dottest(
        Dop,
        *Dop.shape,
        rtol=1e-6,
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Downsample2D_scipy(par):
    """Compare Downsample2D forward with scipy
    gaussian filtering plus subsampling"""
    factors = (
        (par["factors"], par["factors"])
        if isinstance(par["factors"], int)
        else par["factors"]
    )
    sigma = tuple(standard_deviation_from_attenuation(f, 10) for f in factors)

    shape = (par["ny"], par["nx"])
    x = np.random.normal(0.0, 1.0, shape) + par["imag"] * np.random.normal(
        0.0, 1.0, shape
    )
    Dop = Downsample2D(
        (par["ny"], par["nx"]), factors=par["factors"], dtype=par["dtype"]
    )
    y = Dop @ x

    xnp = np.asnumpy(x) if backend == "cupy" else x
    ynp = gaussian_filter(xnp, sigma=sigma, truncate=4.0, mode="constant")[
        :: factors[0], :: factors[1]
    ]
    assert_array_almost_equal(y, np.asarray(ynp), decimal=10)


@pytest.mark.parametrize("par", [(par3)])
def test_Downsample2D_inverse(par):
    """Invert Downsample2D when no decimation is applied (factors=1) as in
    this case the operator is a square, invertible smoothing operator
    """
    x = np.random.normal(0.0, 1.0, (par["ny"], par["nx"]))
    Dop = Downsample2D(
        (par["ny"], par["nx"]), factors=par["factors"], sigma=0.6, dtype=par["dtype"]
    )
    y = Dop @ x
    xinv = lsqr(Dop, y.ravel(), x0=np.zeros(Dop.shape[1]), niter=500, show=0)[0]
    assert_array_almost_equal(x.ravel(), xinv, decimal=3)
