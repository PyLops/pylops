import numpy as np
import pytest

from pylops.basicoperators import Identity
from pylops.utils import dottest
from pylops.utils.seismicevents import hyperbolic2d, hyperbolic3d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.oneway import Deghosting, PhaseShift

np.random.seed(10)

parmod = {
    "ox": -100,
    "dx": 10,
    "nx": 21,
    "oy": -50,
    "dy": 10,
    "ny": 11,
    "ot": 0,
    "dt": 0.004,
    "nt": 50,
    "f0": 40,
}

par1 = {"ny": 8, "nx": 10, "nt": 20, "dtype": "float32"}  # even
par2 = {"ny": 9, "nx": 11, "nt": 21, "dtype": "complex64"}  # odd

# deghosting params
vel_sep = 1000.0  # velocity at separation level
zrec = 20.0  # depth of receivers

# axes and wavelet
t, t2, x, y = makeaxis(parmod)
wav = ricker(t[:41], f0=parmod["f0"])[0]


@pytest.fixture(scope="module")
def create_data2D():
    """Create 2d dataset"""
    t0_plus = np.array([0.02, 0.08])
    t0_minus = t0_plus + 0.04
    vrms = np.array([1400.0, 1800.0])
    amp = np.array([1.0, -0.6])

    p2d_minus = hyperbolic2d(x, t, t0_minus, vrms, amp, wav)[1].T

    kx = np.fft.ifftshift(np.fft.fftfreq(parmod["nx"], parmod["dx"]))
    freq = np.fft.rfftfreq(parmod["nt"], parmod["dt"])

    Pop = -PhaseShift(vel_sep, 2 * zrec, parmod["nt"], freq, kx)

    # Decomposition operator
    Dupop = Identity(parmod["nt"] * parmod["nx"]) + Pop

    p2d = Dupop * p2d_minus.ravel()
    p2d = p2d.reshape(parmod["nt"], parmod["nx"])
    return p2d, p2d_minus


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PhaseShift_2dsignal(par):
    """Dot-test for PhaseShift of 2d signal"""
    vel = 1500.0
    zprop = 200
    freq = np.fft.rfftfreq(par["nt"], 1.0)
    kx = np.fft.fftshift(np.fft.fftfreq(par["nx"], 1.0))

    Pop = PhaseShift(vel, zprop, par["nt"], freq, kx, dtype=par["dtype"])
    assert dottest(Pop, par["nt"] * par["nx"], par["nt"] * par["nx"])


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
        Pop, par["nt"] * par["nx"] * par["ny"], par["nt"] * par["nx"] * par["ny"]
    )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Deghosting_2dsignal(par, create_data2D):
    """Deghosting of 2d data"""
    p2d, p2d_minus = create_data2D

    p2d_minus_inv, p2d_plus_inv = Deghosting(
        p2d,
        parmod["nt"],
        parmod["nx"],
        parmod["dt"],
        parmod["dx"],
        vel_sep,
        zrec,
        win=np.ones_like(p2d),
        npad=0,
        ntaper=0,
        dtype=par["dtype"],
        **dict(damp=1e-10, iter_lim=60)
    )

    print(np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus))
    assert np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus) < 3e-1
