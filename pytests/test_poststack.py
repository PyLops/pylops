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

from pylops.avo.poststack import PoststackInversion, PoststackLinearModelling
from pylops.utils import dottest
from pylops.utils.wavelets import ricker

np.random.seed(10)

# params
dt0 = 0.004
ntwav = 41
nsmooth = 50

# 1d model
nt0 = 201
t0 = np.arange(nt0) * dt0
vp = 1200 + np.arange(nt0) + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 80, nt0))
rho = 1000 + vp + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 30, nt0))

m = np.log(vp * rho)
mback = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m)

# 2d model
inputfile = "testdata/avo/poststack_model.npz"
model = np.load(inputfile)
x, z, m2d = model["x"][::3], model["z"][::3], np.log(model["model"][::3, ::3])
nx, nz = len(x), len(z)

mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m2d, axis=0)
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, mback2d, axis=1)

# stationary wavelet
wav = ricker(t0[: ntwav // 2 + 1], 20)[0]

# non-stationary wavelet
f0s = np.flip(np.arange(nt0) * 0.05 + 3)
wavs = np.array([ricker(t0[:ntwav], f0)[0] for f0 in f0s])
wavc = np.argmax(wavs[0])


par1 = {
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": False,
}  # unregularized
par2 = {
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": False,
    "kind": "centered",
}  # regularized, centered
par3 = {
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": False,
    "kind": "forward",
}  # regularized, forward
par4 = {
    "epsR": None,
    "epsRL1": None,
    "epsI": None,
    "simultaneous": True,
}  # unregularized, simultaneous
par5 = {
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "centered",
}  # regularized, simultaneous, centered
par6 = {
    "epsR": 1e-4,
    "epsRL1": None,
    "epsI": 1e-6,
    "simultaneous": True,
    "kind": "forward",
}  # regularized, simultaneous, forward
par7 = {
    "epsR": 1e-4,
    "epsRL1": 1e-1,
    "epsI": 1e-6,
    "simultaneous": True,
}  # blocky, simultaneous


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PoststackLinearModelling1d(par, dtype, explicit):
    """Dot-test, comparison of dense vs lop implementation and
    inversion for PoststackLinearModelling in 1d with stationary wavelet
    """
    # Dense
    PPop_dense = PoststackLinearModelling(wav.astype(dtype), nt0=nt0, explicit=True)
    assert dottest(
        PPop_dense,
        nt0,
        nt0,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Linear operator
    PPop = PoststackLinearModelling(wav.astype(dtype), nt0=nt0, explicit=False)
    assert dottest(
        PPop, nt0, nt0, rtol=1e-3 if dtype == np.float32 else 1e-6, backend=backend
    )

    # Compare data (dense vs lop) and check dtype
    d = PPop * m.astype(dtype)
    d_dense = PPop_dense * m.astype(dtype)
    madj = PPop.H * d
    madj_dense = PPop_dense.H * d_dense
    assert d.dtype == dtype
    assert d_dense.dtype == dtype
    assert madj.dtype == dtype
    assert madj_dense.dtype == dtype
    assert_array_almost_equal(d, d_dense, decimal=4)
    assert_array_almost_equal(madj, madj_dense, decimal=4)

    # Inversion (note that since the operator is ill-conditioned, we cannot
    # expect a perfect reconstruction, hence the large relative error threshold)
    data = d_dense if explicit else d
    if par["epsR"] is None:
        dict_inv = (
            dict()
            if not explicit
            else dict(rcond=1e-6)
            if int(os.environ.get("TEST_CUPY_PYLOPS", 0))
            else dict(cond=1e-6)
        )  # to avoid instability (cond for scipy abd rcond for cupy)
    else:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=80)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=80)
        )

    data_orig = data.copy()
    minv, _ = PoststackInversion(
        data,
        wav.astype(dtype),
        m0=mback.astype(dtype),
        explicit=explicit,
        epsR=par["epsR"],
        epsI=par["epsI"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    err = 1e-2 if dtype == np.float32 else 8e-3
    assert np.linalg.norm(m - minv) / np.linalg.norm(m) < err

    # Check input data is preserved
    assert_array_equal(data_orig, data)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PoststackLinearModelling1d_nonstationary(par, dtype, explicit):
    """Dot-test, comparison of dense vs lop implementation and
    inversion for PoststackLinearModelling in 1d with nonstationary wavelet
    """
    # Dense
    PPop_dense = PoststackLinearModelling(wavs.astype(dtype), nt0=nt0, explicit=True)
    assert dottest(
        PPop_dense,
        nt0,
        nt0,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Linear operator
    PPop = PoststackLinearModelling(wavs.astype(dtype), nt0=nt0, explicit=False)
    assert dottest(
        PPop, nt0, nt0, rtol=1e-3 if dtype == np.float32 else 1e-6, backend=backend
    )

    # Compare data (dense vs lop) and check dtype
    d = PPop * m.astype(dtype)
    d_dense = PPop_dense * m.astype(dtype)
    madj = PPop.H * d
    madj_dense = PPop_dense.H * d_dense
    assert d.dtype == dtype
    assert d_dense.dtype == dtype
    assert madj.dtype == dtype
    assert madj_dense.dtype == dtype
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    data = d_dense if explicit else d
    if par["epsR"] is None:
        dict_inv = (
            dict()
            if not explicit
            else dict(rcond=1e-6)
            if int(os.environ.get("TEST_CUPY_PYLOPS", 0))
            else dict(cond=1e-6)
        )  # to avoid instability (cond for scipy abd rcond for cupy)
    else:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=80)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=80)
        )

    data_orig = data.copy()
    minv, _ = PoststackInversion(
        data,
        wavs.astype(dtype),
        m0=mback.astype(dtype),
        explicit=explicit,
        epsR=par["epsR"],
        epsI=par["epsI"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    err = 1e-2 if dtype == np.float32 else 8e-3
    assert np.linalg.norm(m - minv) / np.linalg.norm(m) < err

    # Check input data is preserved
    assert_array_equal(data_orig, data)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par6), (par7)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PoststackLinearModelling2d(par, dtype, explicit):
    """Dot-test and inversion for PoststackLinearModelling in 2d"""

    # Dense
    PPop_dense = PoststackLinearModelling(
        wav.astype(dtype), nt0=nz, spatdims=nx, explicit=True
    )
    assert dottest(
        PPop_dense,
        nz * nx,
        nz * nx,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Linear operator
    PPop = PoststackLinearModelling(
        wav.astype(dtype), nt0=nz, spatdims=nx, explicit=False
    )
    assert dottest(
        PPop,
        nz * nx,
        nz * nx,
        rtol=1e-3 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Compare data
    d = PPop * m2d.astype(dtype)
    d_dense = PPop_dense * m2d.astype(dtype)
    madj = PPop.H * d
    madj_dense = PPop_dense.H * d_dense
    assert d.dtype == dtype
    assert d_dense.dtype == dtype
    assert madj.dtype == dtype
    assert madj_dense.dtype == dtype
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    data = d_dense if explicit else d
    if explicit and not par["simultaneous"] and par["epsR"] is None:
        dict_inv = (
            dict()
            if not explicit
            else dict(rcond=1e-6)
            if int(os.environ.get("TEST_CUPY_PYLOPS", 0))
            else dict(cond=1e-6)
        )  # to avoid instability (cond for scipy abd rcond for cupy)
    elif explicit and not par["simultaneous"] and par["epsR"] is not None:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=10)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=10)
        )
    else:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=10)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=10)
        )

    data_orig = data.copy()
    minv2d, _ = PoststackInversion(
        data,
        wav.astype(dtype),
        m0=mback2d.astype(dtype),
        explicit=explicit,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    err = 5e-1 if dtype == np.float32 else 1e-1
    assert np.linalg.norm(m2d - minv2d) / np.linalg.norm(m2d) < err

    # Check input data is preserved
    assert_array_equal(data_orig, data)


@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PoststackInversion1d(par, dtype, explicit):
    """Relative 1d inversion with PoststackInversion checking that
    the input data is preserved
    """
    PPop = PoststackLinearModelling(wav.astype(dtype), nt0=nt0, explicit=explicit)
    data = PPop * m.astype(dtype)
    data_orig = data.copy()

    # Inversion
    if par["epsR"] is None:
        dict_inv = (
            dict()
            if not explicit
            else dict(rcond=1e-6)
            if int(os.environ.get("TEST_CUPY_PYLOPS", 0))
            else dict(cond=1e-6)
        )  # to avoid instability (cond for scipy abd rcond for cupy)
    else:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=80)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=80)
        )

    _ = PoststackInversion(
        data,
        wav.astype(dtype),
        m0=None,
        explicit=explicit,
        epsR=par["epsR"],
        epsI=par["epsI"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    assert_array_equal(data_orig, data)


@pytest.mark.parametrize("par", [(par1), (par4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("explicit", [True, False])
def test_PoststackInversion2d(par, dtype, explicit):
    """Relative 2d inversion with PoststackInversion checking that
    the input data is preserved
    """
    PPop = PoststackLinearModelling(
        wav.astype(dtype), nt0=nz, spatdims=nx, explicit=explicit
    )

    data = PPop * m2d.astype(dtype)
    data_orig = data.copy()

    if explicit and not par["simultaneous"] and par["epsR"] is None:
        dict_inv = (
            dict()
            if not explicit
            else dict(rcond=1e-6)
            if int(os.environ.get("TEST_CUPY_PYLOPS", 0))
            else dict(cond=1e-6)
        )  # to avoid instability (cond for scipy abd rcond for cupy)
    elif explicit and not par["simultaneous"] and par["epsR"] is not None:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=10)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=10)
        )
    else:
        dict_inv = (
            dict(damp=0 if par["epsI"] is None else par["epsI"], iter_lim=10)
            if backend == "numpy"
            else dict(damp=0 if par["epsI"] is None else par["epsI"], niter=10)
        )

    _ = PoststackInversion(
        data,
        wav.astype(dtype),
        m0=None,
        explicit=explicit,
        epsI=par["epsI"],
        epsR=par["epsR"],
        epsRL1=par["epsRL1"],
        simultaneous=par["simultaneous"],
        **dict_inv,
    )
    assert_array_equal(data_orig, data)
