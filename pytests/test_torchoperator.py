import os
import platform

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_equal

    backend = "numpy"
import pytest
import torch

from pylops import (
    MatrixMult,
    Sum,
    TorchOperator,
)
from pylops.utils.backend import to_numpy

par1 = {"ny": 11, "nx": 11}  # square
par2 = {"ny": 21, "nx": 11}  # overdetermined

np.random.seed(0)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator_nd_flatten(par):
    """Check error is raised when nd-array is passed to unbatched operator
    with flatten=True"""
    device = "cpu" if backend == "numpy" else "cuda"
    dims = (par["ny"], par["nx"])
    op = Sum(dims, axis=-1)
    top = TorchOperator(
        op, batch=False, flatten=True, device="cpu" if backend == "numpy" else "gpu"
    )
    x = np.random.normal(0.0, 1.0, dims)
    xt = torch.from_numpy(to_numpy(x)).to(device)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        top(xt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("adjoint", [False, True])
def test_TorchOperator(par, dtype, adjoint):
    """Apply forward/adjoint and gradient. As for linear operators the gradient
    must equal the adjoint of operator applied to the same vector, the two
    results are also checked to be the same.
    """
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype), dtype=dtype
    )
    Top = TorchOperator(
        Dop, batch=False, flatten=True, device="cpu" if backend == "numpy" else "gpu"
    )

    x = np.random.normal(0.0, 1.0, par["ny"] if adjoint else par["nx"]).astype(dtype)
    xt = torch.from_numpy(to_numpy(x)).to(device).view(-1)
    xt.requires_grad = True
    v = np.random.normal(0.0, 1.0, par["nx"] if adjoint else par["ny"]).astype(dtype)
    vt = torch.from_numpy(to_numpy(v)).to(device).view(-1)

    # pylops operator
    if adjoint:
        y = Dop.H * x
        xadj = Dop * v
    else:
        y = Dop * x
        xadj = Dop.H * v

    # torch operator
    yt = Top.adjoint(xt) if adjoint else Top.forward(xt)
    yt.backward(vt)
    yt = yt.detach().cpu().numpy()
    xadjt = xt.grad.cpu().numpy()

    assert yt.dtype == x.dtype
    assert xadjt.dtype == x.dtype
    assert_array_equal(y, yt)
    assert_array_equal(xadj, xadjt)


# @pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("adjoint", [False, True])
def test_TorchOperator_batch(par, dtype, adjoint):
    """Apply forward/adjoint for input with multiple samples (= batch) and flattened arrays"""
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype), dtype=dtype
    )
    Top = TorchOperator(
        Dop, batch=True, flatten=True, device="cpu" if backend == "numpy" else "gpu"
    )

    x = np.random.normal(0.0, 1.0, (4, par["ny"] if adjoint else par["nx"])).astype(
        dtype
    )
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = Dop.rmatmat(x.T).T if adjoint else Dop.matmat(x.T).T
    yt = Top.adjoint(xt) if adjoint else Top.forward(xt)
    yt = yt.detach().cpu().numpy()
    assert yt.dtype == x.dtype
    assert_array_equal(y, yt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("adjoint", [False, True])
def test_TorchOperator_batch_nd(par, dtype, adjoint):
    """Apply forward/adjoint for input with multiple samples (= batch) and nd-arrays"""
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype),
        otherdims=(2,),
        dtype=dtype,
    )
    Top = TorchOperator(
        Dop, batch=True, flatten=False, device="cpu" if backend == "numpy" else "cuda"
    )

    x = np.random.normal(0.0, 1.0, (4, par["ny"] if adjoint else par["nx"], 2)).astype(
        dtype
    )
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = (
        (Dop.H @ x.transpose(1, 2, 0)).transpose(2, 0, 1)
        if adjoint
        else (Dop @ x.transpose(1, 2, 0)).transpose(2, 0, 1)
    )
    yt = Top.adjoint(xt) if adjoint else Top.forward(xt)
    yt = yt.detach().cpu().numpy()

    assert yt.dtype == dtype
    assert_array_equal(y, yt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [par1, par2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("forceflat", [None, False, True])
def test_TorchOperator_different_ndims(
    par,
    dtype,
    batch,
    flatten,
    adjoint,
    forceflat,
):
    """Apply forward/adjoint and gradient when model and data have different
    numbers of dimensions, and check that output shapes depend only on
    ``batch`` and ``flatten`` (not on ``forceflat``)
    """
    device = "cpu" if backend == "numpy" else "cuda"
    op = Sum((par["ny"], par["nx"]), axis=-1, forceflat=forceflat, dtype=dtype)
    Top = TorchOperator(
        op,
        batch=batch,
        flatten=flatten,
        device="cpu" if backend == "numpy" else "gpu",
    )
    input_dims = op.dimsd if adjoint else op.dims
    output_dims = op.dims if adjoint else op.dimsd
    batch_dims = (3,) if batch else ()

    x = np.random.normal(0.0, 1.0, (*batch_dims, *input_dims)).astype(dtype)
    v = np.random.normal(0.0, 1.0, (*batch_dims, *output_dims)).astype(dtype)
    if flatten:
        x = x.reshape(*batch_dims, -1)
        v = v.reshape(*batch_dims, -1)
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True
    vt = torch.from_numpy(to_numpy(v)).to(device)

    # torch reference (forward: sum over last axis;
    # adjoint: spreading over last axis)
    xreft = xt.detach().clone().requires_grad_()
    xrefnd = xreft.reshape(*batch_dims, *input_dims)
    if adjoint:
        yreft = xrefnd.unsqueeze(-1).expand(*batch_dims, *output_dims)
    else:
        yreft = xrefnd.sum(dim=-1)
    if flatten:
        yreft = yreft.reshape(*batch_dims, -1)
    yreft.backward(vt)

    # torch operator
    yt = Top.adjoint(xt) if adjoint else Top.forward(xt)
    yt.backward(vt)

    assert yt.shape == yreft.shape
    assert yt.dtype == xt.dtype
    assert xt.grad.shape == xt.shape
    torch.testing.assert_close(yt, yreft)
    torch.testing.assert_close(xt.grad, xreft.grad)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_TorchOperator_forward_adjoint(par, dtype):
    """Compute gradient of L2 norm (chain of forward and adjoint) and
    compare Jacobian vector product with analytical solution"""
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype), dtype=dtype
    )
    Top = TorchOperator(Dop, batch=False, device="cpu" if backend == "numpy" else "gpu")

    x = np.random.normal(0.0, 1.0, par["nx"]).astype(dtype)
    xt = torch.from_numpy(to_numpy(x)).to(device).view(-1)
    xt.requires_grad = True
    y = -2 * np.arange(par["ny"], dtype=dtype)
    yt = torch.from_numpy(to_numpy(y)).to(device).view(-1)
    v = np.random.normal(0.0, 1.0, par["nx"]).astype(dtype)
    vt = torch.from_numpy(to_numpy(v)).to(device).view(-1)

    # pylops operator
    f = Dop.H * (y - Dop * x)
    jvt = -Dop.H * Dop * v

    # torch operator
    ft = Top.adjoint(yt - Top.forward(xt))
    ft.backward(vt)
    jvtt = xt.grad.cpu().numpy()
    ft = ft.detach().cpu().numpy()

    assert ft.dtype == x.dtype
    assert jvtt.dtype == x.dtype
    assert_array_equal(f, ft)
    assert_array_equal(jvt, jvtt)
