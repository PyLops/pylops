import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import Identity, Restriction
from pylops.optimization.basic import lsqr
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FFT2D, FFTND
from pylops.utils import dottest
from pylops.utils.seismicevents import hyperbolic2d, hyperbolic3d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.oneway import Deghosting, PhaseShift

np.random.seed(10)

# phaseshift params
par1 = {
    "ny": 8,
    "nx": 10,
    "nt": 20,
    "kind": "p",
    "fftengine": "numpy",
    "kwargs_fft": {},
}  # even, p, numpy
par2 = {
    "ny": 9,
    "nx": 11,
    "nt": 21,
    "kind": "p",
    "fftengine": "numpy",
    "kwargs_fft": {},
}  # odd, p, numpy
par1s = {
    "ny": 8,
    "nx": 10,
    "nt": 20,
    "kind": "p",
    "fftengine": "scipy",
    "kwargs_fft": dict(workers=4),
}  # even, p, scipy
par1w = {
    "ny": 8,
    "nx": 10,
    "nt": 20,
    "kind": "p",
    "fftengine": "fft",
    "kwargs_fft": {},
}  # even, p, fftw
par1v = {
    "ny": 8,
    "nx": 10,
    "nt": 20,
    "kind": "vz",
}  # even, vz, numpy
par2v = {
    "ny": 9,
    "nx": 11,
    "nt": 21,
    "kind": "vz",
}  # odd, vz, numpy

# deghosting params
parmod = {
    "ox": -400,
    "dx": 10,
    "nx": 81,
    "oy": -50,
    "dy": 10,
    "ny": 11,
    "ot": 0,
    "dt": 0.004,
    "nt": 100,
    "f0": 40,
}
vel_sep = 1000.0  # velocity at separation level
zrec = 20.0  # depth of receivers

# axes and wavelet
t, t2, x, y = makeaxis(parmod)
wav = ricker(t[:41], f0=parmod["f0"])[0]


def create_data2D(datakind):
    """Create 2d dataset"""
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


def create_data3D(datakind):
    """Create 3d dataset"""
    t0_plus = npp.array([0.02, 0.08])
    t0_minus = t0_plus + 0.04
    vrms = npp.array([1400.0, 1800.0])
    amp = npp.array([1.0, -0.6])

    p3d_minus = hyperbolic3d(x, y, t, t0_minus, vrms, vrms, amp, wav)[1].transpose(
        2, 1, 0
    )

    kx = npp.fft.ifftshift(npp.fft.fftfreq(parmod["nx"], parmod["dx"]))
    ky = npp.fft.ifftshift(npp.fft.fftfreq(parmod["ny"], parmod["dy"]))
    freq = npp.fft.rfftfreq(parmod["nt"], parmod["dt"])

    Pop = -PhaseShift(vel_sep, 2 * zrec, parmod["nt"], freq, kx, ky)

    # Decomposition operator
    Dupop = Identity(parmod["nt"] * parmod["nx"] * parmod["ny"]) + datakind * Pop

    p3d = Dupop * p3d_minus.ravel()
    p3d = p3d.reshape(parmod["nt"], parmod["nx"], parmod["ny"])
    return np.asarray(p3d), np.asarray(p3d_minus)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par1w)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PhaseShift_2dsignal(par, dtype):
    """Dot-test for PhaseShift of 2d signal"""
    vel = 1500.0
    zprop = 200
    freq = np.fft.rfftfreq(par["nt"], 1.0)
    kx = np.fft.fftshift(np.fft.fftfreq(par["nx"], 1.0))

    kwargs_fft = par["kwargs_fft"] if backend == "numpy" else {}
    Pop = PhaseShift(
        vel,
        zprop,
        par["nt"],
        freq,
        kx,
        fftengine=par["fftengine"] if backend == "numpy" else "numpy",
        dtype=dtype,
        **kwargs_fft,
    )
    assert dottest(
        Pop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["nt"], par["nx"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PhaseShift_3dsignal(par, dtype):
    """Dot-test for PhaseShift of 3d signal"""
    vel = 1500.0
    zprop = 200
    freq = np.fft.rfftfreq(par["nt"], 1.0)
    kx = np.fft.fftshift(np.fft.fftfreq(par["nx"], 1.0))
    ky = np.fft.fftshift(np.fft.fftfreq(par["ny"], 1.0))

    Pop = PhaseShift(vel, zprop, par["nt"], freq, kx, ky, dtype=dtype)
    assert dottest(
        Pop,
        par["nt"] * par["nx"] * par["ny"],
        par["nt"] * par["nx"] * par["ny"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.ones((par["nt"], par["nx"], par["ny"]), dtype=dtype)
    y = Pop * x.ravel()
    xadj = Pop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.parametrize("par", [(par1), (par2), (par1v), (par2v)])
def test_Deghosting_2dsignal(par):
    """Deghosting of 2d data"""
    p2d, p2d_minus = create_data2D(1 if par["kind"] == "p" else -1)

    p2d_minus_inv, _ = Deghosting(
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
        dtype=np.float32,
        **dict(damp=1e-10, niter=60),
    )

    assert np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus) < 3e-1


@pytest.mark.parametrize("par", [(par1), (par2), (par1v), (par2v)])
def test_Deghosting_2dsignal_sptransf(par):
    """Deghosting of 2d data with FK sparsifying transform"""
    p2d, p2d_minus = create_data2D(1 if par["kind"] == "p" else -1)

    FOp = FFT2D(
        dims=(parmod["nt"], parmod["nx"]),
        sampling=(parmod["dt"], parmod["dx"]),
        dtype=np.complex128,
    )

    p2d_minus_inv, _ = Deghosting(
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
        sptransf=FOp.H,
        solver=fista,
        dtype=np.float32,
        **dict(eps=1e-4, niter=100),
    )

    assert np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus) < 2e-1


@pytest.mark.parametrize("par", [(par1), (par2), (par1v), (par2v)])
def test_Deghosting_2dsignal_restr(par):
    """Deghosting of 2d data with restriction and FK sparsifying transform"""
    npp.random.seed(10)

    p2d, p2d_minus = create_data2D(1 if par["kind"] == "p" else -1)

    # Restriction operator (90% of available traces)
    nsub = int(npp.round(0.9 * parmod["nx"]))
    iava = np.asarray(npp.sort(npp.random.permutation(parmod["nx"])[:nsub]))
    Rop = Restriction((parmod["nt"], parmod["nx"]), iava, axis=1, dtype=np.complex128)
    p2d_sub = np.real(Rop * p2d.ravel()).reshape(parmod["nt"], nsub)

    FOp = FFT2D(
        dims=(parmod["nt"], parmod["nx"]),
        sampling=(parmod["dt"], parmod["dx"]),
        dtype=np.complex128,
    )

    p2d_minus_inv, p2d_plus_inv = Deghosting(
        p2d_sub,
        parmod["nt"],
        parmod["nx"],
        parmod["dt"],
        parmod["dx"],
        vel_sep,
        zrec,
        kind=par["kind"],
        win=np.ones_like(p2d_sub),
        npad=0,
        ntaper=0,
        restriction=Rop,
        sptransf=FOp.H,
        solver=fista,
        dtype=np.float32,
        **dict(eps=1e-4, niter=100),
    )

    assert p2d_minus_inv.shape == (parmod["nt"], parmod["nx"])
    assert p2d_plus_inv.shape == (parmod["nt"], parmod["nx"])
    assert np.linalg.norm(p2d_minus_inv - p2d_minus) / np.linalg.norm(p2d_minus) < 4e-1


@pytest.mark.parametrize("par", [(par1), (par1v)])
def test_Deghosting_3dsignal(par):
    """Deghosting of 3d data"""
    p3d, p3d_minus = create_data3D(1 if par["kind"] == "p" else -1)

    p3d_minus_inv, _ = Deghosting(
        p3d,
        parmod["nt"],
        (parmod["nx"], parmod["ny"]),
        parmod["dt"],
        (parmod["dx"], parmod["dy"]),
        vel_sep,
        zrec,
        kind=par["kind"],
        win=np.ones_like(p3d),
        npad=(0, 0),
        ntaper=(0, 0),
        solver=lsqr,
        dtype=np.float32,
        **dict(damp=1e-10, niter=60),
    )

    assert np.linalg.norm(p3d_minus_inv - p3d_minus) / np.linalg.norm(p3d_minus) < 3e-1


@pytest.mark.parametrize("par", [(par1), (par1v)])
def test_Deghosting_3dsignal_sptransf(par):
    """Deghosting of 3d data with FK sparsifying transform"""
    p3d, p3d_minus = create_data3D(1 if par["kind"] == "p" else -1)

    FOp = FFTND(
        dims=(parmod["nt"], parmod["nx"], parmod["ny"]),
        sampling=(parmod["dt"], parmod["dx"], parmod["dy"]),
        dtype=np.complex128,
    )

    p3d_minus_inv, _ = Deghosting(
        p3d,
        parmod["nt"],
        (parmod["nx"], parmod["ny"]),
        parmod["dt"],
        (parmod["dx"], parmod["dy"]),
        vel_sep,
        zrec,
        kind=par["kind"],
        win=np.ones_like(p3d),
        npad=(0, 0),
        ntaper=(0, 0),
        sptransf=FOp.H,
        solver=fista,
        dtype=np.float32,
        **dict(eps=1e-4, niter=50),
    )

    assert np.linalg.norm(p3d_minus_inv - p3d_minus) / np.linalg.norm(p3d_minus) < 2e-1


@pytest.mark.parametrize("par", [(par1), (par1v)])
def test_Deghosting_3dsignal_restr(par):
    """Deghosting of 3d data with restriction and FK sparsifying transform"""
    npp.random.seed(10)

    p3d, p3d_minus = create_data3D(1 if par["kind"] == "p" else -1)

    # Restriction operator (90% of available traces along x)
    nsub = int(npp.round(0.9 * parmod["nx"]))
    iava = np.asarray(npp.sort(npp.random.permutation(parmod["nx"])[:nsub]))
    Rop = Restriction(
        (parmod["nt"], parmod["nx"], parmod["ny"]), iava, axis=1, dtype=np.complex128
    )
    p3d_sub = np.real(Rop * p3d.ravel()).reshape(parmod["nt"], nsub, parmod["ny"])

    FOp = FFTND(
        dims=(parmod["nt"], parmod["nx"], parmod["ny"]),
        sampling=(parmod["dt"], parmod["dx"], parmod["dy"]),
        dtype=np.complex128,
    )

    p3d_minus_inv, p3d_plus_inv = Deghosting(
        p3d_sub,
        parmod["nt"],
        (parmod["nx"], parmod["ny"]),
        parmod["dt"],
        (parmod["dx"], parmod["dy"]),
        vel_sep,
        zrec,
        kind=par["kind"],
        win=np.ones_like(p3d_sub),
        npad=(0, 0),
        ntaper=(0, 0),
        restriction=Rop,
        sptransf=FOp.H,
        solver=fista,
        dtype=np.float32,
        **dict(eps=1e-4, niter=50),
    )

    assert p3d_minus_inv.shape == (parmod["nt"], parmod["nx"], parmod["ny"])
    assert p3d_plus_inv.shape == (parmod["nt"], parmod["nx"], parmod["ny"])
    assert np.linalg.norm(p3d_minus_inv - p3d_minus) / np.linalg.norm(p3d_minus) < 4e-1
