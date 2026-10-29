import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal, assert_array_equal
    from cupyx.scipy.signal import filtfilt

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal, assert_array_equal
    from scipy.signal import filtfilt

    backend = "numpy"
import pytest

from pylops.avo.prestack import (
    PrestackInversion,
    PrestackLinearModelling,
    PrestackWaveletModelling,
    _linearizations,
)
from pylops.optimization.basic import lsqr
from pylops.utils import dottest
from pylops.utils.wavelets import ricker

np.random.seed(10)

# params
dt0 = 0.004
ntwav = 41
ntheta = 7
nsmooth = 50

# angles
thetamin, thetamax = 0, 40
theta = np.linspace(thetamin, thetamax, ntheta)

# 1d model
nt0 = 184
t0 = np.arange(nt0) * dt0
vp = 1200 + np.arange(nt0) + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 40, nt0))
vs = 600 + vp / 2 + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 20, nt0))
rho = 1000 + vp + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 30, nt0))

m = np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)
mback = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m, axis=0)


# 2d model
inputfile = "testdata/avo/poststack_model.npz"
model = np.load(inputfile)
z, x, model = (
    model["z"][::3] / 1000.0,
    model["x"][::5] / 1000.0,
    1000 * model["model"][::3, ::5],
)
nx, nz = len(x), len(z)

mvp = model.copy()
mvs = model / 2
mrho = model / 3 + 300

m2d = np.log(np.stack((mvp, mvs, mrho), axis=1))
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m2d, axis=0)
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, mback2d, axis=2)

# wavelet
wav, twav, wavc = ricker(t0[: ntwav // 2 + 1], 20)

# shifted wavelet
wavoff = 10
wav_phase = np.hstack((wav[wavoff:], np.zeros(wavoff)))


# constant vsvp, aki-richards approx, unregularized, trace-by-trace
par1 = {
    "vsvp": 0.5,
    "linearization": "akirich",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": False,
    "kind": "centered",
}
# constant vsvp, fatti approx, unregularized, trace-by-trace
par2 = {
    "vsvp": 0.5,
    "linearization": "fatti",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": False,
    "kind": "forward",
}
# time-variant vsvp, aki-richards approx, unregularized, trace-by-trace
par3 = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "akirich",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": False,
    "kind": "centered",
}
# time-variant  vsvp, fatti approx, unregularized, trace-by-trace
par4 = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "fatti",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": False,
    "kind": "forward",
}

# constant vsvp, aki-richards approx, unregularized, simultaneous
par1s = {
    "vsvp": 0.5,
    "linearization": "akirich",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": True,
    "kind": "centered",
}
# constant vsvp, fatti approx, unregularized, simultaneous
par2s = {
    "vsvp": 0.5,
    "linearization": "fatti",
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": True,
    "kind": "forward",
}
# time-variant vsvp, aki-richards approx, unregularized, simultaneous
par3s = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "akirich",
    "epsR": None,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "centered",
}
# time-variant  vsvp, fatti approx, unregularized, simultaneous
par4s = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "fatti",
    "epsR": None,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "forward",
}

# constant vsvp, aki-richards approx, regularized, simultaneous
par1r = {
    "vsvp": 0.5,
    "linearization": "akirich",
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "centered",
}
# constant vsvp, fatti approx, regularized, simultaneous
par2r = {
    "vsvp": 0.5,
    "linearization": "fatti",
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "forward",
}
# time-variant vsvp, aki-richards approx, regularized, simultaneous
par3r = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "akirich",
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "centered",
}
# time-variant  vsvp, fatti approx, regularized, simultaneous
par4r = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "fatti",
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "forward",
}

# constant vsvp, aki-richards approx, blocky, simultaneous
par1b = {
    "vsvp": 0.5,
    "linearization": "akirich",
    "epsR": 1e-4,
    "epsRL1": 1e-2,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "centered",
}
# time-variant vsvp, aki-richards approx, regularized, simultaneous
par3b = {
    "vsvp": np.linspace(0.4, 0.6, nt0),
    "linearization": "akirich",
    "epsR": 1e-4,
    "epsRL1": 1e-2,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "forward",
}

# small problems (1d, 2d, 3d) to compare explicit and chained regularized inversion
parord1 = {"nt": 50, "nm": 3, "spatdims": (), "dtype": "float64"}
parord2 = dict(parord1, spatdims=(4,))
parord3 = dict(parord1, spatdims=(4, 3))


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par4),
        (par1s),
        (par2s),
        (par3s),
        (par4s),
        (par1r),
        (par2r),
        (par3r),
        (par4r),
        (par1b),
        (par3b),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PrestackLinearModelling(par, dtype, explicit):
    """Dot-test, comparison of dense vs lop implementation and
    inversion for PrestackLinearModelling
    """
    # Dense
    PPop_dense = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nt0,
        linearization=par["linearization"],
        explicit=True,
        kind=par["kind"],
    )
    assert dottest(
        PPop_dense,
        nt0 * ntheta,
        nt0 * _linearizations[par["linearization"]],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Linear operator
    PPop = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nt0,
        linearization=par["linearization"],
        explicit=False,
        kind=par["kind"],
    )
    assert dottest(
        PPop,
        nt0 * ntheta,
        nt0 * _linearizations[par["linearization"]],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Compare data
    d = PPop * m.astype(dtype).ravel()
    d = d.reshape(nt0, ntheta)
    d_dense = PPop_dense * m.astype(dtype).T.ravel()
    d_dense = d_dense.reshape(ntheta, nt0).T
    madj = PPop.H * d.ravel()
    madj = madj.reshape(nt0, _linearizations[par["linearization"]])
    madj_dense = PPop_dense.H * d_dense.T.ravel()
    madj_dense = madj_dense.reshape(_linearizations[par["linearization"]], nt0).T
    assert d.dtype == dtype
    assert d_dense.dtype == dtype
    assert madj.dtype == dtype
    assert madj_dense.dtype == dtype
    assert_array_almost_equal(d, d_dense, decimal=4)
    assert_array_almost_equal(madj, madj_dense, decimal=4)

    # Inversion
    dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if not par["simultaneous"]:
        dict_inv = {}
    if not explicit:
        dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if par["epsRL1"] is not None:
        dict_inv = (
            dict(mu=0.1, niter_outer=5, niter_inner=5, iter_lim=8)
            if backend == "numpy"
            else dict(mu=0.1, niter_outer=5, niter_inner=5, niter=8)
        )

    d_orig = d.copy()
    minv, _ = PrestackInversion(
        d,
        theta.astype(dtype),
        wav.astype(dtype),
        m0=mback.astype(dtype),
        explicit=explicit,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        kind=par["kind"],
        returnres=True,
        **dict_inv,
    )
    err = 1e-1 if dtype == np.float32 else 5e-2
    assert np.linalg.norm(m - minv) / np.linalg.norm(minv) < err

    # Check input data is preserved
    assert_array_equal(d_orig, d)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PrestackWaveletModelling(par, dtype):
    """Dot-test and inversion for PrestackWaveletModelling"""
    # Operators
    Wavestop = PrestackWaveletModelling(
        m.astype(dtype),
        theta.astype(dtype),
        nwav=ntwav,
        wavc=wavc,
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        linearization=par["linearization"],
    )
    assert dottest(
        Wavestop,
        nt0 * ntheta,
        ntwav,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    Wavestop_phase = PrestackWaveletModelling(
        m.astype(dtype),
        theta.astype(dtype),
        nwav=ntwav,
        wavc=wavc,
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        linearization=par["linearization"],
    )
    assert dottest(
        Wavestop_phase,
        nt0 * ntheta,
        ntwav,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Create data
    d = (Wavestop * wav).reshape(ntheta, nt0).T
    d_phase = (Wavestop_phase * wav_phase).reshape(ntheta, nt0).T
    assert d.dtype == dtype
    assert d_phase.dtype == dtype

    # Estimate wavelet
    wav_est = lsqr(
        Wavestop,
        d.T.ravel(),
        x0=np.zeros_like(wav),
        damp=1e-20,
        niter=300,
        atol=0,
        btol=0,
        conlim=np.inf,
        show=0,
    )[0]
    wav_phase_est = lsqr(
        Wavestop_phase,
        d_phase.T.ravel(),
        x0=np.zeros_like(wav_phase),
        damp=1e-20,
        niter=300,
        atol=0,
        btol=0,
        conlim=np.inf,
        show=0,
    )[0]

    assert_array_almost_equal(wav, wav_est, decimal=3)
    assert_array_almost_equal(wav_phase, wav_phase_est, decimal=3)


@pytest.mark.parametrize(
    "par", [(par1), (par3), (par2s), (par4s), (par1r), (par3r), (par1b), (par3b)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PrestackLinearModelling2d(par, dtype, explicit):
    """Dot-test and inversion for PoststackLinearModelling in 2d"""
    nm = _linearizations[par["linearization"]]
    # Dense
    PPop_dense = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nz,
        spatdims=(nx,),
        linearization=par["linearization"],
        explicit=True,
    )
    assert dottest(
        PPop_dense,
        nz * ntheta * nx,
        nz * nm * nx,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Linear operator
    PPop = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nz,
        spatdims=(nx,),
        linearization=par["linearization"],
        explicit=False,
    )
    assert dottest(
        PPop_dense,
        nz * ntheta * nx,
        nz * nm * nx,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Compare data
    d = (PPop * m2d.astype(dtype).ravel()).reshape(nz, ntheta, nx)
    d_dense = (
        (PPop_dense * m2d.astype(dtype).swapaxes(0, 1).ravel())
        .reshape(ntheta, nz, nx)
        .swapaxes(0, 1)
    )
    assert d.dtype == dtype
    assert d_dense.dtype == dtype
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if not par["simultaneous"]:
        dict_inv = {}
    if not explicit:
        dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if par["epsRL1"] is not None:
        dict_inv = (
            dict(mu=0.1, niter_outer=5, niter_inner=5, iter_lim=8)
            if backend == "numpy"
            else dict(mu=0.1, niter_outer=5, niter_inner=5, niter=8)
        )
    minv2d = PrestackInversion(
        d,
        theta.astype(dtype),
        wav.astype(dtype),
        m0=mback2d.astype(dtype),
        explicit=explicit,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    err = 1e-1 if dtype == np.float32 else 5e-2
    assert np.linalg.norm(m2d - minv2d) / np.linalg.norm(minv2d) < err


@pytest.mark.parametrize("par", [(parord1), (parord2), (parord3)])
@pytest.mark.parametrize("epsI", [0.1, [0.1, 0.2, 0.3]])
@pytest.mark.parametrize("background", [False, True])
def test_PrestackInversion_l2_explicit_chained(par, epsI, background):
    """Explicit and chained PrestackInversion with L2 regularizers agree"""
    shape = (par["nt"], par["nm"], *par["spatdims"])
    PPop = PrestackLinearModelling(
        wav, theta, nt0=par["nt"], spatdims=par["spatdims"] or None
    )
    nd = par["nt"] * ntheta * int(np.prod(np.array(par["spatdims"] or (1,))))
    nmod = par["nt"] * par["nm"] * int(np.prod(np.array(par["spatdims"] or (1,))))
    assert dottest(PPop, nd, nmod, backend=backend)

    mord = np.random.normal(size=shape).astype(par["dtype"])
    dord = (PPop @ mord.ravel()).reshape(par["nt"], ntheta, *par["spatdims"])
    m0 = np.random.normal(size=shape).astype(par["dtype"]) if background else None
    dict_inv = (
        dict(iter_lim=1000, atol=1e-11, btol=1e-11)
        if backend == "numpy"
        else dict(niter=1000, tol=1e-20)
    )
    dict_inv = dict(m0=m0, simultaneous=True, epsR=0.4, epsI=epsI, **dict_inv)
    chained = PrestackInversion(dord.copy(), theta, wav, explicit=False, **dict_inv)
    explicit = PrestackInversion(dord.copy(), theta, wav, explicit=True, **dict_inv)
    assert_array_almost_equal(explicit, chained, decimal=7)


@pytest.mark.parametrize("par", [(parord1), (parord2), (parord3)])
@pytest.mark.parametrize("epsI", [0.1, [0.1, 0.2, 0.3]])
def test_PrestackInversion_l1_explicit_chained(par, epsI):
    """Explicit and chained PrestackInversion with L1 regularizers agree"""
    shape = (par["nt"], par["nm"], *par["spatdims"])
    PPop = PrestackLinearModelling(
        wav, theta, nt0=par["nt"], spatdims=par["spatdims"] or None
    )
    mord = np.random.normal(size=shape).astype(par["dtype"])
    dord = (PPop @ mord.ravel()).reshape(par["nt"], ntheta, *par["spatdims"])
    dict_inv = (
        dict(iter_lim=100, atol=1e-11, btol=1e-11)
        if backend == "numpy"
        else dict(niter=100, tol=1e-20)
    )
    dict_inv = dict(
        epsR=0.4,
        epsI=epsI,
        epsRL1=0.1,
        mu=0.1,
        niter_outer=8,
        niter_inner=5,
        **dict_inv,
    )
    chained = PrestackInversion(dord.copy(), theta, wav, explicit=False, **dict_inv)
    explicit = PrestackInversion(dord.copy(), theta, wav, explicit=True, **dict_inv)
    assert_array_almost_equal(explicit, chained, decimal=7)

@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PrestackLinearInversion1d(par, dtype):
    """Relative 1d inversion with PrestackLinearModelling checking that
    the input data is preserved
    """
    PPop = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nt0,
        linearization=par["linearization"],
        explicit=False,
        kind=par["kind"],
    )
    d = PPop * m.astype(dtype).ravel()
    d = d.reshape(nt0, ntheta)
    d_orig = d.copy()

    dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if not par["simultaneous"]:
        dict_inv = {}
    if par["epsRL1"] is not None:
        dict_inv = (
            dict(mu=0.1, niter_outer=5, niter_inner=5, iter_lim=8)
            if backend == "numpy"
            else dict(mu=0.1, niter_outer=5, niter_inner=5, niter=8)
        )

    _ = PrestackInversion(
        d,
        theta.astype(dtype),
        wav.astype(dtype),
        m0=None,
        explicit=False,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        kind=par["kind"],
        returnres=True,
        **dict_inv,
    )
    assert_array_equal(d_orig, d)


@pytest.mark.parametrize("par", [(par1), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PrestackLinearInversion2d(par, dtype):
    """Relative 2d inversion with PrestackLinearModelling checking that
    the input data is preserved
    """
    PPop = PrestackLinearModelling(
        wav.astype(dtype),
        theta.astype(dtype),
        vsvp=par["vsvp"]
        if isinstance(par["vsvp"], float)
        else par["vsvp"].astype(dtype),
        nt0=nz,
        spatdims=(nx,),
        linearization=par["linearization"],
        explicit=False,
    )

    d = (PPop * m2d.astype(dtype).ravel()).reshape(nz, ntheta, nx)
    d_orig = d.copy()

    dict_inv = dict(iter_lim=10) if backend == "numpy" else dict(niter=10)
    if par["epsRL1"] is not None:
        dict_inv = (
            dict(mu=0.1, niter_outer=5, niter_inner=5, iter_lim=8)
            if backend == "numpy"
            else dict(mu=0.1, niter_outer=5, niter_inner=5, niter=8)
        )
    _ = PrestackInversion(
        d,
        theta.astype(dtype),
        wav.astype(dtype),
        m0=mback2d.astype(dtype),
        explicit=False,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        returnres=True,
        **dict_inv,
    )
    assert_array_equal(d_orig, d)
