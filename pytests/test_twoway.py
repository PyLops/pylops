import devito
import numpy as np

from pylops.utils import dottest
from pylops.waveeqprocessing.twoway import AcousticWave2D

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
