import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import pytest

from pylops.avo.prestack import PrestackInversion, PrestackLinearModelling
from pylops.utils import dottest

par1 = {
    "nt": 31,
    "ntheta": 4,
    "nx": 2,
    "ny": 3,
    "nwav": (3, 5, 7, 5),
    "dtype": "float32",
}
par2 = {
    "nt": 31,
    "ntheta": 4,
    "nx": 2,
    "ny": 3,
    "nwav": (3, 33, 5, 7),
    "dtype": "float64",
}
par3 = {
    "nt": 31,
    "ntheta": 4,
    "nx": 2,
    "ny": 3,
    "nwav": (4, 6, 8, 2),
    "dtype": "float64",
}

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("bad", ["empty", "count", "rank", "empty_wavelet"])
def test_PrestackLinearModelling_angle_wavelets_invalid(par, explicit, bad):
    """Validate the angle wavelet sequence."""
    theta = np.linspace(5, 40, par["ntheta"], dtype=par["dtype"])
    wav = [np.random.normal(size=n).astype(par["dtype"]) for n in par["nwav"]]
    if bad == "empty":
        wav = []
    elif bad == "count":
        wav = wav[:-1]
    elif bad == "rank":
        wav[0] = wav[0][:, None]
    else:
        wav[0] = wav[0][:0]
    with pytest.raises(ValueError, match="wavelet"):
        PrestackLinearModelling(wav, theta, nt0=par["nt"], explicit=explicit)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("kind", ["centered", "forward"])
@pytest.mark.parametrize(
    "linearization",
    [
        "akirich",
        "fatti",
        pytest.param(
            "ps",
            marks=pytest.mark.skipif(
                backend == "cupy",
                reason="PS coefficients do not support CuPy vsvp arrays",
            ),
        ),
    ],
)
@pytest.mark.parametrize("container", [list, tuple])
def test_PrestackLinearModelling_angle_wavelets(
    par, ndim, kind, linearization, container
):
    """Dot-test and compare dense and matrix-free angle wavelets."""
    spatdims = (
        None if ndim == 1 else (par["nx"],) if ndim == 2 else (par["nx"], par["ny"])
    )
    nspat = 1 if ndim == 1 else par["nx"] if ndim == 2 else par["nx"] * par["ny"]
    theta = np.linspace(5, 40, par["ntheta"], dtype=par["dtype"])
    wav = container(np.random.normal(size=n).astype(par["dtype"]) for n in par["nwav"])
    vsvp = np.linspace(0.4, 0.6, par["nt"], dtype=par["dtype"])
    operators = [
        PrestackLinearModelling(
            wav,
            theta,
            vsvp=vsvp,
            spatdims=spatdims,
            linearization=linearization,
            kind=kind,
            explicit=explicit,
        )
        for explicit in (False, True)
    ]
    for op in operators:
        assert dottest(
            op,
            par["nt"] * par["ntheta"] * nspat,
            par["nt"] * 3 * nspat,
            rtol=1e-4 if par["dtype"] == "float32" else 1e-6,
            # Float32 inner products can nearly cancel even when the forward
            # and adjoint vectors agree with their independent references.
            atol=1e-5 if par["dtype"] == "float32" else 1e-21,
            backend=backend,
        )
    model = np.random.normal(size=(par["nt"], 3, nspat)).astype(par["dtype"])
    data = (operators[0] * model.ravel()).reshape(par["nt"], par["ntheta"], nspat)
    dense = (
        (operators[1] * model.swapaxes(0, 1).ravel())
        .reshape(par["ntheta"], par["nt"], nspat)
        .swapaxes(0, 1)
    )
    assert_array_almost_equal(data, dense, decimal=4)
    assert data.dtype == par["dtype"]
    if linearization in ("akirich", "fatti"):
        # Independent reflectivity and full-convolution reference. Comparing
        # dense and chained operators alone can miss a shared centering error.
        dm = np.zeros_like(model)
        if kind == "centered":
            dm[1:-1] = (model[2:] - model[:-2]) / 2
        else:
            dm[:-1] = model[1:] - model[:-1]
        angle = np.deg2rad(theta)[None, :, None]
        ratio = vsvp[:, None, None]
        density = (
            0.5 - 2 * ratio**2 * np.sin(angle) ** 2
            if linearization == "akirich"
            else 0.5 * (4 * ratio**2 * np.sin(angle) ** 2 - np.tan(angle) ** 2)
        )
        reflectivity = (
            dm[:, 0, None, :] / (2 * np.cos(angle) ** 2)
            - 4 * ratio**2 * np.sin(angle) ** 2 * dm[:, 1, None, :]
            + density * dm[:, 2, None, :]
        )
        expected = np.empty_like(data)
        for j, w in enumerate(wav):
            center = len(w) // 2
            for k in range(nspat):
                full = np.convolve(reflectivity[:, j, k], w, mode="full")
                expected[:, j, k] = full[center : center + par["nt"]]
        assert_array_almost_equal(data, expected, decimal=4)
    adj = (operators[0].H * data.ravel()).reshape(par["nt"], 3, nspat)
    dense_adj = (
        (operators[1].H * data.swapaxes(0, 1).ravel())
        .reshape(3, par["nt"], nspat)
        .swapaxes(0, 1)
    )
    assert_array_almost_equal(
        adj, dense_adj, decimal=3 if par["dtype"] == "float32" else 6
    )


@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("explicit", [False, True])
def test_PrestackInversion_repeated_angle_wavelet(par, explicit):
    """Preserve single-wavelet modelling and inversion with repeated wavelets."""
    theta = np.linspace(5, 40, par["ntheta"], dtype=par["dtype"])
    wav = np.random.normal(size=par["nwav"][0]).astype(par["dtype"])
    waves = [wav] * par["ntheta"]
    model = np.random.normal(size=(par["nt"], 3)).astype(par["dtype"])
    op = PrestackLinearModelling(wav, theta, nt0=par["nt"])
    data = (op * model.ravel()).reshape(par["nt"], par["ntheta"])
    options = {} if explicit else {"iter_lim" if backend == "numpy" else "niter": 10}
    single = PrestackInversion(data.copy(), theta, wav, explicit=explicit, **options)
    repeated = PrestackInversion(
        data.copy(), theta, waves, explicit=explicit, **options
    )
    assert_array_almost_equal(single, repeated, decimal=3)
