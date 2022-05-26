import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.utils.seismicevents import linear2d, linear3d, makeaxis
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.mdd import MDC, MDD

PAR = {
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

# nt odd, single-sided, full fft
par1 = PAR.copy()
par1["twosided"] = False
par1["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2))

# nt odd, double-sided, full fft
par2 = PAR.copy()
par2["twosided"] = True
par2["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2))

# nt odd, single-sided, truncated fft
par3 = PAR.copy()
par3["twosided"] = False
par3["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2)) - 30

# nt odd, double-sided, truncated fft
par4 = PAR.copy()
par4["twosided"] = True
par4["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2)) - 30

# nt even, single-sided, full fft
par5 = PAR.copy()
par5["nt"] -= 1
par5["twosided"] = False
par5["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2))

# nt even, double-sided, full fft
par6 = PAR.copy()
par6["nt"] -= 1
par6["twosided"] = True
par6["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2))

# nt even, single-sided, truncated fft
par7 = PAR.copy()
par7["nt"] -= 1
par7["twosided"] = False
par7["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2)) - 30

# nt even, double-sided, truncated fft
par8 = PAR.copy()
par8["nt"] -= 1
par8["twosided"] = True
par8["nfmax"] = int(np.ceil((PAR["nt"] + 1.0) / 2)) - 30


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_MDC_1virtualsource(par):
    """Dot-test and inversion for MDC operator of 1 virtual source"""
    if par["twosided"]:
        par["nt2"] = 2 * par["nt"] - 1
    else:
        par["nt2"] = par["nt"]
    v = 1500
    it0_m = 25
    t0_m = it0_m * par["dt"]
    theta_m = 0
    amp_m = 1.0

    it0_G = np.array([25, 50, 75])
    t0_G = it0_G * par["dt"]
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1.0, 0.6, 2.0)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par["f0"])[0]

    # Generate model
    _, mwav = linear2d(x, t, v, t0_m, theta_m, amp_m, wav)
    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par["twosided"]:
        mwav = np.concatenate((np.zeros((par["nx"], par["nt"] - 1)), mwav), axis=-1)
        Gwav = np.concatenate(
            (np.zeros((par["ny"], par["nx"], par["nt"] - 1)), Gwav), axis=-1
        )

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par["nt2"], axis=-1)
    Gwav_fft = Gwav_fft[..., : par["nfmax"]]

    MDCop = MDC(
        Gwav_fft.transpose(2, 0, 1),
        nt=par["nt2"],
        nv=1,
        dt=par["dt"],
        dr=par["dx"],
        twosided=par["twosided"],
    )
    dottest(MDCop, par["nt2"] * par["ny"], par["nt2"] * par["nx"])
    mwav = mwav.T
    d = MDCop * mwav.ravel()
    d = d.reshape(par["nt2"], par["ny"])

    for it, amp in zip(it0_G, amp_G):
        ittot = it0_m + it
        if par["twosided"]:
            ittot += par["nt"] - 1
        assert (
            np.abs(
                d[ittot, par["ny"] // 2]
                - np.abs(wav**2).sum()
                * amp_m
                * amp
                * par["nx"]
                * par["dx"]
                * par["dt"]
                * np.sqrt(par["nt2"])
            )
            < 1e-2
        )

    minv = MDD(
        Gwav[:, :, par["nt"] - 1 :] if par["twosided"] else Gwav,
        d[par["nt"] - 1 :].T if par["twosided"] else d.T,
        dt=par["dt"],
        dr=par["dx"],
        nfmax=par["nfmax"],
        twosided=par["twosided"],
        add_negative=True,
        adjoint=False,
        psf=False,
        dottest=False,
        **dict(damp=1e-10, iter_lim=50, show=0)
    )
    assert_array_almost_equal(mwav, minv.T, decimal=2)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7), (par8)]
)
def test_MDC_Nvirtualsources(par):
    """Dot-test and inversion for MDC operator of N virtual source"""
    if par["twosided"]:
        par["nt2"] = 2 * par["nt"] - 1
    else:
        par["nt2"] = par["nt"]
    v = 1500
    it0_m = 25
    t0_m = it0_m * par["dt"]
    theta_m = 0
    phi_m = 0
    amp_m = 1.0

    it0_G = np.array([25, 50, 75])
    t0_G = it0_G * par["dt"]
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1.0, 0.6, 2.0)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par["f0"])[0]

    # Generate model
    _, mwav = linear3d(x, x, t, v, t0_m, theta_m, phi_m, amp_m, wav)

    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par["twosided"]:
        mwav = np.concatenate(
            (np.zeros((par["nx"], par["nx"], par["nt"] - 1)), mwav), axis=-1
        )
        Gwav = np.concatenate(
            (np.zeros((par["ny"], par["nx"], par["nt"] - 1)), Gwav), axis=-1
        )

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par["nt2"], axis=-1)
    Gwav_fft = Gwav_fft[..., : par["nfmax"]]

    MDCop = MDC(
        Gwav_fft.transpose(2, 0, 1),
        nt=par["nt2"],
        nv=par["nx"],
        dt=par["dt"],
        dr=par["dx"],
        twosided=par["twosided"],
    )
    dottest(
        MDCop, par["nt2"] * par["ny"] * par["nx"], par["nt2"] * par["nx"] * par["nx"]
    )

    mwav = mwav.transpose(2, 0, 1)
    d = MDCop * mwav.ravel()
    d = d.reshape(par["nt2"], par["ny"], par["nx"])

    for it, amp in zip(it0_G, amp_G):
        ittot = it0_m + it
        if par["twosided"]:
            ittot += par["nt"] - 1
        assert (
            d[ittot, par["ny"] // 2, par["nx"] // 2]
            > d[ittot - 1, par["ny"] // 2, par["nx"] // 2]
        )
        assert (
            d[ittot, par["ny"] // 2, par["nx"] // 2]
            > d[ittot + 1, par["ny"] // 2, par["nx"] // 2]
        )

    minv = MDD(
        Gwav[:, :, par["nt"] - 1 :] if par["twosided"] else Gwav,
        d[par["nt"] - 1 :].transpose(1, 2, 0)
        if par["twosided"]
        else d.transpose(1, 2, 0),
        dt=par["dt"],
        dr=par["dx"],
        nfmax=par["nfmax"],
        twosided=par["twosided"],
        add_negative=True,
        adjoint=False,
        psf=False,
        dottest=False,
        **dict(damp=1e-10, iter_lim=50, show=0)
    )
    assert_array_almost_equal(mwav, minv.transpose(2, 0, 1), decimal=2)
