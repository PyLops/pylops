import os

import numpy as np
import pytest

from pylops.signalprocessing import UDCT
from pylops.utils import dottest

par1 = {"ny": 16, "nx": 20, "nz": 16, "imag": 0}  # real
par1j = {"ny": 16, "nx": 20, "nz": 16, "imag": 1j}  # complex


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par1j)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_UDCT2D(par, dtype):
    """Dot test for UDCT Operator to 2D inputs"""
    np.random.seed(0)
    cdtype = (np.empty(0, dtype=dtype) + 1j * np.empty(0, dtype=dtype)).dtype
    x = np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype) + par[
        "imag"
    ] * np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype)

    Uop = UDCT(
        (par["ny"], par["nx"]),
        transform_kind="real" if par["imag"] == 0 else "complex",
        dtype=cdtype,
    )
    assert dottest(
        Uop,
        complexflag=2 if par["imag"] == 0 else 3,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Uop.H * (Uop * x)
    np.testing.assert_allclose(x, y, rtol=5e-4 if dtype == np.float32 else 1e-6)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par1j)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_UDCT3D(par, dtype):
    """Dot test for UDCT Operator to 3D inputs"""
    np.random.seed(0)
    cdtype = (np.empty(0, dtype=dtype) + 1j * np.empty(0, dtype=dtype)).dtype
    x = np.random.normal(0.0, 1.0, (par["ny"], par["nx"], par["nz"])).astype(
        dtype
    ) + par["imag"] * np.random.normal(
        0.0, 1.0, (par["ny"], par["nx"], par["nz"])
    ).astype(dtype)

    Uop = UDCT(
        (par["ny"], par["nx"], par["nz"]),
        transform_kind="real" if par["imag"] == 0 else "complex",
        dtype=cdtype,
    )
    assert dottest(
        Uop,
        complexflag=2 if par["imag"] == 0 else 3,
        rtol=5e-4 if dtype == np.float32 else 1e-6,
    )

    y = Uop.H * (Uop * x)
    np.testing.assert_allclose(x, y, rtol=5e-2 if dtype == np.float32 else 1e-6)
