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

np.random.seed(10)
par1 = {"nt": 9, "nm": 3, "ntheta": 5, "spatdims": (), "dtype": "float64"}
par2 = dict(par1, spatdims=(4,))
par3 = dict(par1, spatdims=(4, 3))


def _problem(par):
    shape = (par["nt"], par["nm"], *par["spatdims"])
    theta = np.linspace(0, 35, par["ntheta"]).astype(par["dtype"])
    wav = np.array([0.2, 1.0, 0.2], dtype=par["dtype"])
    model = np.random.normal(size=shape).astype(par["dtype"])
    op = PrestackLinearModelling(
        wav, theta, nt0=par["nt"], spatdims=par["spatdims"] or None
    )
    data = (op @ model.ravel()).reshape(par["nt"], par["ntheta"], *par["spatdims"])
    return shape, theta, wav, op, data


@pytest.mark.parametrize("par", [par1, par2, par3])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("epsI", [0.1, [0.1, 0.2, 0.3]])
@pytest.mark.parametrize("background", [False, True])
def test_PrestackInversion_l2_model_order(par, explicit, epsI, background):
    """Recover the independent augmented least-squares solution in model order."""
    shape, theta, wav, op, data = _problem(par)
    assert dottest(op, op.shape[0], op.shape[1], backend=backend)
    size = op.shape[1]
    # Build the finite-difference matrix without PyLops regularizers.
    basis = np.eye(size).reshape(*shape, size)
    regularizer = np.zeros_like(basis)
    axes = (0,) if not par["spatdims"] else (0, 2) if len(shape) == 3 else (2, 3)
    for axis in axes:
        center = [slice(None)] * basis.ndim
        left = center.copy()
        right = center.copy()
        center[axis] = slice(1, -1)
        left[axis] = slice(None, -2)
        right[axis] = slice(2, None)
        regularizer[tuple(center)] += (
            basis[tuple(left)] - 2 * basis[tuple(center)] + basis[tuple(right)]
        )
    weights = np.array(epsI).reshape(
        (1, par["nm"], *([1] * len(par["spatdims"]))) if isinstance(epsI, list) else ()
    )
    weights = np.broadcast_to(weights, shape).ravel()
    matrix = np.vstack(
        [op.todense(), 0.4 * regularizer.reshape(size, size), np.diag(weights)]
    )
    rhs = np.concatenate([data.ravel(), np.zeros(2 * size)])
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0].reshape(shape)
    m0 = np.random.normal(size=shape).astype(par["dtype"]) if background else None
    original = data.copy()
    kwargs = (
        dict(iter_lim=1000, atol=1e-11, btol=1e-11)
        if backend == "numpy"
        else dict(niter=1000, tol=1e-20)
    )
    actual, residual = PrestackInversion(
        data,
        theta,
        wav,
        m0=m0,
        explicit=explicit,
        simultaneous=True,
        epsR=0.4,
        epsI=epsI,
        returnres=True,
        **kwargs,
    )
    assert_array_almost_equal(actual, expected, decimal=7)
    assert_array_almost_equal(
        residual, (data.ravel() - op @ actual.ravel()).reshape(data.shape), decimal=7
    )
    assert_array_almost_equal(data, original, decimal=12)


@pytest.mark.parametrize("par", [par1, par2, par3])
@pytest.mark.parametrize("epsI", [0.1, [0.1, 0.2, 0.3]])
def test_PrestackInversion_l1_model_order(par, epsI):
    """Keep split-Bregman regularizers consistent across explicit layouts."""
    _, theta, wav, _, data = _problem(par)
    kwargs = (
        dict(iter_lim=100, atol=1e-11, btol=1e-11)
        if backend == "numpy"
        else dict(niter=100, tol=1e-20)
    )
    common = dict(
        epsR=0.4, epsI=epsI, epsRL1=0.1, mu=0.1, niter_outer=8, niter_inner=5, **kwargs
    )
    chained = PrestackInversion(data.copy(), theta, wav, explicit=False, **common)
    explicit = PrestackInversion(data.copy(), theta, wav, explicit=True, **common)
    assert_array_almost_equal(explicit, chained, decimal=7)
