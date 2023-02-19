import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

PAR = {
    "ny": 3,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nsy": 4,
    "nry": 3,
    "nsx": 6,
    "nrx": 2,
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
par3 = {"mode": "byot", "dynamic": False}
par1d = {"mode": "analytic", "dynamic": True}
par2d = {"mode": "eikonal", "dynamic": True}
par3d = {"mode": "byot", "dynamic": True}


def test_identify_geometry():
    """Identify geometry, check expected outputs"""
    # 2d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s2d, r2d)
    assert ndims == 2
    assert shiftdim == 0
    assert [1, 2] == [1, 2]
    assert list(dims) == [PAR["nx"], PAR["nz"]]
    assert ny == 1
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsx"]
    assert nr == PAR["nrx"]
    assert list(dsamp) == [dx, dz]
    assert list(origin) == [0, 0]

    # 3d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s3d, r3d, y=y)
    assert ndims == 3
    assert shiftdim == 1
    assert list(dims) == [PAR["ny"], PAR["nx"], PAR["nz"]]
    assert ny == PAR["ny"]
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsy"] * PAR["nsx"]
    assert nr == PAR["nry"] * PAR["nrx"]
    assert list(dsamp) == [dy, dx, dz]
    assert list(origin) == [0, 0, 0]


def test_traveltime_ana():
    """Check analytical traveltimes in homogenous medium for horizontal and
    vertical paths
    """
    src = np.array([100, 0])[:, np.newaxis]

    (
        trav_srcs_ana,
        trav_recs_ana,
        dist_srcs_ana,
        dist_recs_ana,
        _,
        _,
    ) = Kirchhoff._traveltime_table(
        np.arange(0, 200, 1), np.arange(0, 200, 1), src, src, v0, mode="analytic"
    )
    assert dist_srcs_ana[0, 0] + dist_recs_ana[0, 0] == 200
    assert trav_srcs_ana[0, 0] == 100 / v0
    assert trav_recs_ana[0, 0] == 100 / v0


def test_traveltime_table():
    """Compare analytical and eikonal traveltimes in homogenous medium"""
    if skfmm_enabled:
        # 2d
        (
            trav_ana,
            trav_srcs_ana,
            trav_recs_ana,
            dist_ana,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s2d, r2d, v0, mode="analytic")

        (
            trav_eik,
            trav_srcs_eik,
            trav_recs_eik,
            dist_eik,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0 * np.ones((PAR["nx"], PAR["nz"])), mode="eikonal"
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_ana, decimal=2)
        assert_array_almost_equal(trav_ana, trav_eik, decimal=2)

        # 3d
        (
            trav_srcs_ana,
            trav_recs_ana,
            dist_srcs_ana,
            dist_recs_ana,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s3d, r3d, v0, y=y, mode="analytic")

        (
            trav_srcs_eik,
            trav_recs_eik,
            dist_srcs_eik,
            dist_recs_eik,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z,
            x,
            s3d,
            r3d,
            v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"])),
            y=y,
            mode="eikonal",
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_eik, decimal=2)
        assert_array_almost_equal(trav_ana, trav_eik, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1d), (par2d), (par3d)])
def test_kirchhoff2d(par):
    """Dot-test for Kirchhoff operator"""
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]))

    if par["mode"] == "byot":
        trav, _, _, dist, _, _ = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0, mode="analytic"
        )
        amp = 1 / (dist + 1e-2 * dist.max())
    else:
        trav = None
        amp = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z,
            x,
            t,
            s2d,
            r2d,
            vel if par["mode"] == "eikonal" else v0,
            wav,
            wavc,
            y=None,
            trav=trav,
            amp=amp,
            mode=par["mode"],
        )
        assert dottest(Dop, PAR["nsx"] * PAR["nrx"] * PAR["nt"], PAR["nz"] * PAR["nx"])


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_kirchhoff3d(par):
    """Dot-test for Kirchhoff operator"""
    vel = v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"]))

    if par["mode"] == "byot":
        trav, _, _, dist, _, _ = Kirchhoff._traveltime_table(
            z, x, s3d, r3d, v0, y=y, mode="analytic"
        )
        amp = 1 / (dist + 1e-2 * dist.max())
    else:
        trav = None
        amp = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z,
            x,
            t,
            s3d,
            r3d,
            vel if par["mode"] == "eikonal" else v0,
            wav,
            wavc,
            y=y,
            trav=trav,
            amp=amp,
            mode=par["mode"],
        )
        assert dottest(
            Dop,
            PAR["nsx"] * PAR["nrx"] * PAR["nsy"] * PAR["nry"] * PAR["nt"],
            PAR["nz"] * PAR["nx"] * PAR["ny"],
        )


@pytest.mark.parametrize("par", [(par1), (par1d)])
def test_kirchhoff2d_trav_vs_travsrcrec(par):
    """Compare Kirchhoff operator forward and adjoint when using trav (original behavior)
    or trav_src and trav_rec (new reccomended behaviour)"""
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]))

    # new behaviour
    Dop = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav,
        wavc,
        y=None,
        mode=par["mode"],
    )

    # old behaviour
    trav = Dop.trav_srcs.reshape(
        PAR["nx"] * PAR["nz"], PAR["nsx"], 1
    ) + Dop.trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
    trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])

    amp = None
    if par["dynamic"]:
        amp = Dop.amp_srcs.reshape(
            PAR["nx"] * PAR["nz"], PAR["nsx"], 1
        ) + Dop.amp_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
        amp = amp.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])

    D1op = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav,
        wavc,
        y=None,
        trav=trav,
        amp=amp,
        mode=par["mode"],
    )

    # forward
    xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"])
    assert_array_almost_equal(Dop @ xx, D1op @ xx, decimal=2)

    # adjoint
    yy = np.random.normal(0, 1, PAR["nrx"] * PAR["nsx"] * PAR["nt"])
    assert_array_almost_equal(Dop.H @ yy, D1op.H @ yy, decimal=2)
