import os

import numpy as np
import pytest

from pylops.signalprocessing import DCT
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11}
par2 = {"ny": 11, "nx": 21}
par3 = {"ny": 20, "nx": 20}


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
def test_DCT1D(par, dtype, dct_type):
    """Dot test for Discrete Cosine Transform Operator 1D"""

    t = np.arange(par["ny"], dtype=dtype) + 1.0

    Dct = DCT(dims=(par["ny"],), type=dct_type, dtype=dtype)
    assert dottest(
        Dct,
        par["ny"],
        par["ny"],
        complexflag=0,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Dct.H * (Dct * t)
    np.testing.assert_allclose(t, y, rtol=5e-4 if dtype == np.float32 else 1e-6)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
@pytest.mark.parametrize("axes", [0, 1])
def test_DCT2D(par, dtype, dct_type, axes):
    """Dot test for Discrete Cosine Transform Operator 2D"""

    t = np.outer(
        np.arange(par["ny"], dtype=dtype) + 1.0, np.arange(par["nx"], dtype=dtype) + 1.0
    )

    Dct = DCT(dims=t.shape, type=dct_type, axes=axes, dtype=dtype)
    assert dottest(
        Dct,
        par["nx"] * par["ny"],
        par["nx"] * par["ny"],
        complexflag=0,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Dct.H * (Dct * t)
    np.testing.assert_allclose(t, y, rtol=5e-4 if dtype == np.float32 else 1e-6)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
@pytest.mark.parametrize("axes", [0, 1, 2])
def test_DCT3D(par, dtype, dct_type, axes):
    """Dot test for Discrete Cosine Transform Operator 3D"""

    t = np.random.rand(par["nx"], par["nx"], par["nx"])

    Dct = DCT(dims=t.shape, type=dct_type, axes=axes, dtype=dtype)
    assert dottest(
        Dct,
        par["nx"] * par["nx"] * par["nx"],
        par["nx"] * par["nx"] * par["nx"],
        complexflag=0,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Dct.H * (Dct * t)
    np.testing.assert_allclose(t, y, rtol=5e-4 if dtype == np.float32 else 1e-6)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_DCT_workers(par, dtype):
    """Dot test for Discrete Cosine Transform Operator with workers"""
    t = np.arange(par["ny"]) + 1

    Dct = DCT(dims=(par["ny"],), type=1, dtype=dtype, workers=2)
    assert dottest(
        Dct,
        par["ny"],
        par["ny"],
        complexflag=0,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Dct.H * (Dct * t)
    np.testing.assert_allclose(t, y, rtol=5e-4 if dtype == np.float32 else 1e-6)
