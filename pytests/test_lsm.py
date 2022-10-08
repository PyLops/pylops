import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.lsm import LSM

PAR = {
    "ny": 10,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nsy": 4,
    "nry": 8,
    "nsx": 6,
    "nrx": 4,
}

# Check if skfmm is available and by-pass tests using it otherwise. This is
# currently required for Travis as since we moved to Python3.8 it has
# stopped working
try:
    import skfmm  # noqa: F401

    skfmm_enabled = True
except ImportError:
    skfmm_enabled = False

v0 = 500
y = np.arange(PAR["ny"]) * PAR["dy"]
x = np.arange(PAR["nx"]) * PAR["dx"]
z = np.arange(PAR["nz"]) * PAR["dz"]
t = np.arange(PAR["nt"]) * PAR["dt"]

sy = np.linspace(y.min(), y.max(), PAR["nsy"])
sx = np.linspace(x.min(), x.max(), PAR["nsx"])
syy, sxx = np.meshgrid(sy, sx, indexing="ij")
s2d = np.vstack((sx, 2 * np.ones(PAR["nsx"])))
s3d = np.vstack((syy.ravel(), sxx.ravel(), 2 * np.ones(PAR["nsx"] * PAR["nsy"])))

ry = np.linspace(y.min(), y.max(), PAR["nry"])
rx = np.linspace(x.min(), x.max(), PAR["nrx"])
ryy, rxx = np.meshgrid(ry, rx, indexing="ij")
r2d = np.vstack((rx, 2 * np.ones(PAR["nrx"])))
r3d = np.vstack((ryy.ravel(), rxx.ravel(), 2 * np.ones(PAR["nrx"] * PAR["nry"])))

wav, _, wavc = ricker(t[:41], f0=40)

par1 = {"mode": "analytic", "dynamic": False}
par2 = {"mode": "eikonal", "dynamic": False}
par1d = {"mode": "analytic", "dynamic": True}
par2d = {"mode": "eikonal", "dynamic": True}


def test_unknown_mode():
    """Check error is raised if unknown mode is passed"""
    with pytest.raises(NotImplementedError):
        _ = LSM(z, x, t, s2d, r2d, 0, np.ones(3), 1, mode="foo")


@pytest.mark.parametrize("par", [(par1), (par2), (par1d), (par2d)])
def test_lsm2d(par):
    """Dot-test and inverse for LSM operator"""
    if skfmm_enabled or par["mode"] != "eikonal":
        vel = v0 * np.ones((PAR["nx"], PAR["nz"]))
        refl = np.zeros((PAR["nx"], PAR["nz"]))
        refl[:, PAR["nz"] // 2] = 1
        refl[:, 3 * PAR["nz"] // 4] = 1

        lsm = LSM(
            z,
            x,
            t,
            s2d,
            r2d,
            vel if par["mode"] == "eikonal" else v0,
            wav,
            wavc,
            mode=par["mode"],
            dynamic=par["dynamic"],
            dottest=True,
        )

        d = lsm.Demop * refl.ravel()
        d = d.reshape(PAR["nsx"], PAR["nrx"], PAR["nt"])

        minv = lsm.solve(d.ravel(), **dict(iter_lim=100, show=True))
        minv = minv.reshape(PAR["nx"], PAR["nz"])

        dinv = lsm.Demop * minv.ravel()
        dinv = dinv.reshape(PAR["nsx"], PAR["nrx"], PAR["nt"])

        assert_array_almost_equal(d / d.max(), dinv / d.max(), decimal=2)
        assert_array_almost_equal(refl / refl.max(), minv / refl.max(), decimal=1)
