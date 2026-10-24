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

from pylops import MatrixMult, Sum, TorchOperator
from pylops.utils.backend import to_numpy

par1 = {"ny": 11, "nx": 11, "dims": (2, 3)}  # square
par2 = {"ny": 21, "nx": 11, "dims": (2, 3, 4)}  # overdetermined

np.random.seed(0)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_TorchOperator(par, dtype):
    """Apply forward and gradient. As for linear operators the gradient
    must equal the adjoint of operator applied to the same vector, the two
    results are also checked to be the same.
    """
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype), dtype=dtype
    )
    Top = TorchOperator(Dop, batch=False, device="cpu" if backend == "numpy" else "gpu")

    x = np.random.normal(0.0, 1.0, par["nx"]).astype(dtype)
    xt = torch.from_numpy(to_numpy(x)).to(device).view(-1)
    xt.requires_grad = True
    v = np.random.normal(0.0, 1.0, par["ny"]).astype(dtype)
    vt = torch.from_numpy(to_numpy(v)).to(device).view(-1)

    # pylops operator
    y = Dop * x
    xadj = Dop.H * v

    # torch operator
    yt = Top.apply(xt)
    yt.backward(vt, retain_graph=True)
    yt = yt.detach().cpu().numpy()
    xadjt = xt.grad.cpu().numpy()

    assert yt.dtype == x.dtype
    assert xadjt.dtype == x.dtype
    assert_array_equal(y, yt)
    assert_array_equal(xadj, xadjt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_TorchOperator_batch(par, dtype):
    """Apply forward for input with multiple samples (= batch) and flattened arrays"""
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype), dtype=dtype
    )
    Top = TorchOperator(Dop, batch=True, device="cpu" if backend == "numpy" else "gpu")

    x = np.random.normal(0.0, 1.0, (4, par["nx"])).astype(dtype)
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = Dop.matmat(x.T).T
    yt = Top.apply(xt)
    yt = yt.detach().cpu().numpy()
    assert yt.dtype == x.dtype
    assert_array_equal(y, yt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_TorchOperator_batch_nd(par, dtype):
    """Apply forward for input with multiple samples (= batch) and nd-arrays"""
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(dtype),
        otherdims=(2,),
        dtype=dtype,
    )
    Top = TorchOperator(
        Dop, batch=True, flatten=False, device="cpu" if backend == "numpy" else "cuda"
    )

    x = np.random.normal(0.0, 1.0, (4, par["nx"], 2)).astype(dtype)
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = (Dop @ x.transpose(1, 2, 0)).transpose(2, 0, 1)
    yt = Top.apply(xt)
    yt = yt.detach().cpu().numpy()

    assert yt.dtype == dtype
    assert_array_equal(y, yt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1)])
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
    v = np.random.normal(0.0, 1.0, par["ny"]).astype(dtype)
    vt = torch.from_numpy(to_numpy(v)).to(device).view(-1)

    # pylops operator
    f = Dop.H * (y - Dop * x)
    jvt = -Dop.H * Dop * v

    # torch operator
    ft = Top.adjoint(yt - Top.forward(xt))
    ft.backward(vt, retain_graph=True)
    jvtt = xt.grad.cpu().numpy()
    ft = ft.detach().cpu().numpy()

    assert ft.dtype == x.dtype
    assert jvtt.dtype == x.dtype
    assert_array_equal(f, ft)
    assert_array_equal(jvt, jvtt)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [par1, par2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_TorchOperator_batch_changes_rank(par, dtype, flatten, adjoint):
    """Preserve values and gradients when batched model/data ranks differ."""
    device = "cpu" if backend == "numpy" else "cuda"
    op = Sum(par["dims"], axis=-1, forceflat=False, dtype=dtype)
    top = TorchOperator(op, batch=True, flatten=flatten, device=device)
    input_dims = op.dimsd if adjoint else op.dims
    output_dims = op.dims if adjoint else op.dimsd
    input_shape = (
        (3, int(np.prod(np.array(input_dims)))) if flatten else (3, *input_dims)
    )
    values = np.arange(int(np.prod(np.array(input_shape))), dtype=dtype).reshape(
        input_shape
    )
    values = torch.from_numpy(to_numpy(values)).to(device)
    actual_input = (values / 20).requires_grad_()
    reference_input = actual_input.detach().clone().requires_grad_()
    shaped = reference_input.reshape(3, *input_dims)
    if adjoint:
        reference = shaped.unsqueeze(-1).expand(3, *output_dims)
        result = top.adjoint(actual_input)
    else:
        reference = shaped.sum(dim=-1)
        result = top.forward(actual_input)
    if flatten:
        reference = reference.reshape(3, -1)

    torch.testing.assert_close(result, reference)
    assert result.shape == reference.shape
    weights = torch.arange(
        1, result.numel() + 1, dtype=result.dtype, device=device
    ).reshape(result.shape)
    (result * weights).sum().backward()
    (reference * weights).sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
