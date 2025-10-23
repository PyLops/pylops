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

from pylops.utils import dottest
from pylops.utils.backend import to_numpy
from pylops.utils.seismicevents import linear2d, linear3d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.mdd import MDC, MDD

# Modelling parameters
parmod = {
    "ox": 0,
    "dx": 2,
    "nx": 10,
    "oy": 0,
    "dy": 2,
    "ny": 20,
    "ot": 0,
    "dt": 0.004,
    "nt": 401,
    "f0": 20,
}

# Other parameters
v = 1500

it0_m = 25
theta_m = 0
phi_m = 0
amp_m = 1.0
t0_m = it0_m * parmod["dt"]

it0_G = npp.array([25, 50, 75])
theta_G = (0, 0, 0)
phi_G = (0, 0, 0)
amp_G = (1.0, 0.6, 2.0)
t0_G = it0_G * parmod["dt"]

# Test parameters
# nt odd, single-sided, full fft
par1 = parmod.copy()
par1["twosided"] = False
par1["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2))

# nt odd, double-sided, full fft
par2 = parmod.copy()
par2["twosided"] = True
par2["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2))

# nt odd, single-sided, truncated fft
par3 = parmod.copy()
par3["twosided"] = False
par3["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2)) - 30

# nt odd, double-sided, truncated fft
par4 = parmod.copy()
par4["twosided"] = True
par4["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2)) - 30

# nt even, single-sided, full fft
par5 = parmod.copy()
par5["nt"] -= 1
par5["twosided"] = False
par5["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2))

# nt even, double-sided, full fft
par6 = parmod.copy()
par6["nt"] -= 1
par6["twosided"] = True
par6["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2))

# nt even, single-sided, truncated fft
par7 = parmod.copy()
par7["nt"] -= 1
par7["twosided"] = False
par7["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2)) - 30

# nt even, double-sided, truncated fft
par8 = parmod.copy()
par8["nt"] -= 1
par8["twosided"] = True
par8["nfmax"] = int(npp.ceil((parmod["nt"] + 1.0) / 2)) - 30


def create_data(par, nv):
    """Create dataset"""
    if par["twosided"]:
        nt2 = 2 * par["nt"] - 1
    else:
        nt2 = par["nt"]

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=parmod["f0"])[0]

    # Generate model
    if nv == 1:
        _, mwav = linear2d(x, t, v, t0_m, theta_m, amp_m, wav)
    else:
        _, mwav = linear3d(x, x, t, v, t0_m, theta_m, phi_m, amp_m, wav)

    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par["twosided"]:
        if nv == 1:
            mwav = npp.concatenate(
                (npp.zeros((parmod["nx"], par["nt"] - 1)), mwav), axis=-1
            )
        else:
            mwav = npp.concatenate(
                (npp.zeros((parmod["nx"], parmod["nx"], par["nt"] - 1)), mwav), axis=-1
            )
        Gwav = npp.concatenate(
            (npp.zeros((parmod["ny"], parmod["nx"], par["nt"] - 1)), Gwav), axis=-1
        )

    # Define MDC linear operator
    Gwav_fft = npp.fft.fft(Gwav, nt2, axis=-1)
    Gwav_fft = Gwav_fft[..., : par["nfmax"]]

    return nt2, wav, mwav, Gwav, Gwav_fft


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_MDC_1virtualsource(par):
    """Dot-test and inversion for MDC operator of 1 virtual source"""
    nt2, wav, mwav, Gwav, Gwav_fft = create_data(par, 1)

    MDCop = MDC(
        np.asarray(Gwav_fft).transpose(2, 0, 1),
        nt=nt2,
        nv=1,
        dt=parmod["dt"],
        dr=parmod["dx"],
        twosided=par["twosided"],
    )
    dottest(MDCop, nt2 * parmod["ny"], nt2 * parmod["nx"], backend=backend)
    mwav = np.asarray(mwav).T
    d = MDCop * mwav.ravel()
    d = d.reshape(nt2, parmod["ny"])

    for it, amp in zip(it0_G, amp_G):
        ittot = it0_m + it
        if par["twosided"]:
            ittot += par["nt"] - 1
        assert (
            npp.abs(
                to_numpy(d[ittot, parmod["ny"] // 2])
                - npp.abs(wav**2).sum()
                * amp_m
                * amp
                * parmod["nx"]
                * parmod["dx"]
                * parmod["dt"]
                * npp.sqrt(nt2)
            )
            < 1e-2
        )

    solver_dict = (
        dict(damp=1e-10, iter_lim=50)
        if backend == "numpy"
        else dict(damp=1e-10, niter=50)
    )
    minv = MDD(
        np.asarray(Gwav[:, :, par["nt"] - 1 :])
        if par["twosided"]
        else np.asarray(Gwav),
        d[par["nt"] - 1 :].T if par["twosided"] else d.T,
        dt=parmod["dt"],
        dr=parmod["dx"],
        nfmax=par["nfmax"],
        twosided=par["twosided"],
        add_negative=True,
        adjoint=False,
        psf=False,
        dottest=False,
        **solver_dict,
    )
    assert_array_almost_equal(mwav, minv.T, decimal=2)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_MDC_Nvirtualsources(par):
    """Dot-test and inversion for MDC operator of N virtual source"""
    nt2, _, mwav, Gwav, Gwav_fft = create_data(par, parmod["nx"])

    MDCop = MDC(
        np.asarray(Gwav_fft).transpose(2, 0, 1),
        nt=nt2,
        nv=parmod["nx"],
        dt=parmod["dt"],
        dr=parmod["dx"],
        twosided=par["twosided"],
    )
    dottest(
        MDCop,
        nt2 * parmod["ny"] * parmod["nx"],
        nt2 * parmod["nx"] * parmod["nx"],
        backend=backend,
    )

    mwav = np.asarray(mwav).transpose(2, 0, 1)
    d = MDCop * mwav.ravel()
    d = d.reshape(nt2, parmod["ny"], parmod["nx"])

    for it, _ in zip(it0_G, amp_G):
        ittot = it0_m + it
        if par["twosided"]:
            ittot += par["nt"] - 1
        assert (
            d[ittot, parmod["ny"] // 2, parmod["nx"] // 2]
            > d[ittot - 1, parmod["ny"] // 2, parmod["nx"] // 2]
        )
        assert (
            d[ittot, parmod["ny"] // 2, parmod["nx"] // 2]
            > d[ittot + 1, parmod["ny"] // 2, parmod["nx"] // 2]
        )

    solver_dict = (
        dict(damp=1e-10, iter_lim=50)
        if backend == "numpy"
        else dict(damp=1e-10, niter=50)
    )
    minv = MDD(
        np.asarray(Gwav[:, :, par["nt"] - 1 :])
        if par["twosided"]
        else np.asarray(Gwav),
        d[par["nt"] - 1 :].transpose(1, 2, 0)
        if par["twosided"]
        else d.transpose(1, 2, 0),
        dt=parmod["dt"],
        dr=parmod["dx"],
        nfmax=par["nfmax"],
        twosided=par["twosided"],
        add_negative=True,
        adjoint=False,
        psf=False,
        dottest=False,
        **solver_dict,
    )
    assert_array_almost_equal(mwav, minv.transpose(2, 0, 1), decimal=2)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="SciPy engine not compatible with CuPy",
)
@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_MDC_1virtualsource_scipy(par):
    """Dot-test for MDC operator of 1 virtual source with scipy engine and workers"""
    nt2, _, _, _, Gwav_fft = create_data(par, 1)

    MDCop = MDC(
        np.asarray(Gwav_fft).transpose(2, 0, 1),
        nt=nt2,
        nv=1,
        dt=parmod["dt"],
        dr=parmod["dx"],
        twosided=par["twosided"],
        engine="scipy",
        **dict(workers=4),
    )
    dottest(MDCop, nt2 * parmod["ny"], nt2 * parmod["nx"], backend=backend)
