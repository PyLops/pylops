import os

import numpy as np
import pytest

from pylops.utils import deps, dottest
from pylops.waveeqprocessing.twoway import AcousticWave2D

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    import devito

    devito.configuration["log-level"] = "ERROR"


par = {
    "ny": 10,
    "nx": 12,
    "nz": 20,
    "tn": 500,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "nr": 8,
    "ns": 2,
}

v0 = 2
y = np.arange(par["ny"]) * par["dy"]
x = np.arange(par["nx"]) * par["dx"]
z = np.arange(par["nz"]) * par["dz"]

sx = np.linspace(x.min(), x.max(), par["ns"])
rx = np.linspace(x.min(), x.max(), par["nr"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_acwave2d():
    """Dot-test for AcousticWave2D operator"""
    Dop = AcousticWave2D(
        (par["nx"], par["nz"]),
        (0, 0),
        (par["dx"], par["dz"]),
        np.ones((par["nx"], par["nz"])) * 2e3,
        sx,
        5,
        rx,
        5,
        0.0,
        par["tn"],
        "Ricker",
        space_order=4,
        nbl=30,
        f0=15,
        dtype="float32",
    )

    assert dottest(
        Dop, par["ns"] * par["nr"] * Dop.geometry.nt, par["nz"] * par["nx"], atol=1e-1
    )
