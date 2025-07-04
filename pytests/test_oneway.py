import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import Identity
from pylops.optimization.basic import lsqr
from pylops.utils import dottest
from pylops.utils.seismicevents import hyperbolic2d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.oneway import Deghosting, PhaseShift

np.random.seed(10)

parmod = {
    "ox": -400,
    "dx": 10,
    "nx": 81,
    "oy": -50,
    "dy": 10,
    "ny": 11,
    "ot": 0,
    "dt": 0.004,
    "nt": 50,
    "f0": 40,
}

par1 = {"ny": 8, "nx": 10, "nt": 20, "kind": "p", "dtype": "float32"}  # even, p
par2 = {"ny": 9, "nx": 11, "nt": 21, "kind": "p", "dtype": "float32"}  # odd, p
par1v = {"ny": 8, "nx": 10, "nt": 20, "kind": "vz", "dtype": "float32"}  # even, vz
par2v = {"ny": 9, "nx": 11, "nt": 21, "kind": "vz", "dtype": "float32"}  # odd, vz

# deghosting params
vel_sep = 1000.0  # velocity at separation level
zrec = 20.0  # depth of receivers

# axes and wavelet
t, t2, x, y = makeaxis(parmod)
wav = ricker(t[:41], f0=parmod["f0"])[0]


@pytest.fixture
def create_data2D():
    """Create 2d dataset"""

    def core(datakind):
        t0_plus = npp.array([0.02, 0.08])
        t0_minus = t0_plus + 0.04
        vrms = npp.array([1400.0, 1800.0])
        amp = npp.array([1.0, -0.6])

        p2d_minus = hyperbolic2d(x, t, t0_minus, vrms, amp, wav)[1].T

        kx = npp.fft.ifftshift(npp.fft.fftfreq(parmod["nx"], parmod["dx"]))
        freq = npp.fft.rfftfreq(parmod["nt"], parmod["dt"])

        Pop = -PhaseShift(vel_sep, 2 * zrec, parmod["nt"], freq, kx)

        # Decomposition operator
        Dupop = Identity(parmod["nt"] * parmod["nx"]) + datakind * Pop

        p2d = Dupop * p2d_minus.ravel()
        p2d = p2d.reshape(parmod["nt"], parmod["nx"])
        return np.asarray(p2d), np.asarray(p2d_minus)

    return core


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PhaseShift_2dsignal(par):
    """Dot-test for PhaseShift of 2d signal"""
    vel = 1500.0
    zprop = 200
    freq = np.fft.rfftfreq(par["nt"], 1.0)
    kx = np.fft.fftshift(np.fft.fftfreq(par["nx"], 1.0))

    Pop = PhaseShift(vel, zprop, par["nt"], freq, kx, dtype=par["dtype"])
    assert dottest(
        Pop, par["nt"] * par["nx"], par["nt"] * par["nx"], rtol=1e-3, backend=backend
    )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PhaseShift_3dsignal(par):
    """Dot-test for PhaseShift of 3d signal"""
    vel = 1500.0
    zprop = 200
    freq = np.fft.rfftfreq(par["nt"], 1.0)
    kx = np.fft.fftshift(np.fft.fftfreq(par["nx"], 1.0))
    ky = np.fft.fftshift(np.fft.fftfreq(par["ny"], 1.0))

    Pop = PhaseShift(vel, zprop, par["nt"], freq, kx, ky, dtype=par["dtype"])
    assert dottest(
        Pop,
        par["nt"] * par["nx"] * par["ny"],
        par["nt"] * par["nx"] * par["ny"],
        rtol=1e-3,
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par1), (par2), (par1v), (par2v)])
def test_Deghosting_2dsignal(par, create_data2D):
    """Deghosting of 2d data"""
    p2d, p2d_minus = create_data2D(1 if par["kind"] == "p" else -1)

    p2d_minus_inv, p2d_plus_inv = Deghosting(
        p2d,
        parmod["nt"],
        parmod["nx"],
        parmod["dt"],
        parmod["dx"],
        vel_sep,
        zrec,
        kind=par["kind"],
        win=np.ones_like(p2d),
        npad=0,
        ntaper=0,
        solver=lsqr,
        dtype=par["dtype"],
        **dict(damp=1e-10, niter=60),
    )

    assert np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus) < 3e-1
