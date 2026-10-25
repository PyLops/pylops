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

from pylops.avo.poststack import PoststackInversion, PoststackLinearModelling
from pylops.avo.prestack import PrestackInversion, PrestackLinearModelling
from pylops.utils import dottest

np.random.seed(10)
par1 = {"nt": 9, "nm": 3, "ntheta": 5, "spatdims": (), "dtype": "float64"}
par2 = dict(par1, spatdims=(4,))
par3 = dict(par1, spatdims=(4, 3))
par1f = dict(par1, dtype="float32")


def _problem(par, prestack):
    wav = np.array([0.2, 1.0, 0.2], dtype=par["dtype"])
    common = dict(nt0=par["nt"], spatdims=par["spatdims"] or None)
    if prestack:
        theta = np.linspace(0, 35, par["ntheta"]).astype(par["dtype"])
        op = PrestackLinearModelling(wav, theta, **common)
        model_shape = (par["nt"], par["nm"], *par["spatdims"])
        data_shape = (par["nt"], par["ntheta"], *par["spatdims"])
        args = (theta, wav)
    else:
        op = PoststackLinearModelling(wav, **common)
        model_shape = data_shape = (par["nt"], *par["spatdims"])
        args = (wav,)
    model = np.random.normal(size=model_shape).astype(par["dtype"])
    data = (op @ model.ravel()).reshape(data_shape)
    return op, model_shape, data, args


def _invert(data, args, prestack, **kwargs):
    solver = dict(iter_lim=200) if backend == "numpy" else dict(niter=200)
    if prestack:
        return PrestackInversion(data, *args, returnres=True, **kwargs, **solver)
    return PoststackInversion(data, *args, **kwargs, **solver)


@pytest.mark.skipif(backend == "cupy", reason="NumPy read-only arrays")
@pytest.mark.parametrize("prestack", [False, True])
def test_Inversion_readonly_residual_input(prestack):
    """Accept read-only input data when returning residuals."""
    _, _, data, args = _problem(par1, prestack)
    data.setflags(write=False)
    _invert(data, args, prestack)
    assert not data.flags.writeable


@pytest.mark.parametrize("par", [par1, par2, par3, par1f])
@pytest.mark.parametrize("prestack", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_Inversion_residual_preserves_input(par, prestack, explicit, background):
    """Preserve caller data and return reproducible models and residuals."""
    op, shape, data, args = _problem(par, prestack)
    assert dottest(
        op,
        op.shape[0],
        op.shape[1],
        backend=backend,
        rtol=1e-3 if par["dtype"] == "float32" else 1e-6,
    )
    original = data.copy()
    m0 = np.random.normal(size=shape).astype(par["dtype"]) if background else None
    kwargs = dict(m0=m0, explicit=explicit, simultaneous=True)
    model, residual = _invert(data, args, prestack, **kwargs)
    assert_array_almost_equal(data, original, decimal=12)
    assert residual.dtype == original.dtype
    expected_residual = (original.ravel() - op @ model.ravel()).astype(original.dtype)
    assert_array_almost_equal(
        residual.ravel(),
        expected_residual,
        decimal=5 if par["dtype"] == "float32" else 10,
    )
    repeated_model, repeated_residual = _invert(data, args, prestack, **kwargs)
    assert_array_almost_equal(
        repeated_model, model, decimal=5 if par["dtype"] == "float32" else 10
    )
    assert_array_almost_equal(
        repeated_residual, residual, decimal=5 if par["dtype"] == "float32" else 10
    )
