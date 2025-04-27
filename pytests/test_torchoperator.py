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
import numpy as npp
import pytest
import torch

from pylops import MatrixMult, TorchOperator
from pylops.utils.backend import to_numpy

par1 = {"ny": 11, "nx": 11, "dtype": np.float32}  # square
par2 = {"ny": 21, "nx": 11, "dtype": np.float32}  # overdetermined

np.random.seed(0)


@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator(par):
    """Apply forward and gradient. As for linear operators the gradient
    must equal the adjoint of operator applied to the same vector, the two
    results are also checked to be the same.
    """
    # temporarily, skip tests on mac as torch seems not to recognized
    # numpy when v2 is installed
    if platform.system() == "Darwin":
        return
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(np.random.normal(0.0, 1.0, (par["ny"], par["nx"])))
    Top = TorchOperator(Dop, batch=False, device="cpu" if backend == "numpy" else "gpu")

    x = np.random.normal(0.0, 1.0, par["nx"])
    xt = torch.from_numpy(to_numpy(x)).to(device).view(-1)
    xt.requires_grad = True
    v = np.random.normal(0.0, 1.0, par["ny"])
    vt = torch.from_numpy(to_numpy(v)).to(device).view(-1)

    # pylops operator
    y = Dop * x
    xadj = Dop.H * v

    # torch operator
    yt = Top.apply(xt)
    yt.backward(vt, retain_graph=True)

    assert_array_equal(y, yt.detach().cpu().numpy())
    assert_array_equal(xadj, xt.grad.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator_batch(par):
    """Apply forward for input with multiple samples (= batch) and flattened arrays"""
    # temporarily, skip tests on mac as torch seems not to recognized
    # numpy when v2 is installed
    if platform.system() == "Darwin":
        return
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(np.random.normal(0.0, 1.0, (par["ny"], par["nx"])))
    Top = TorchOperator(Dop, batch=True, device="cpu" if backend == "numpy" else "gpu")

    x = np.random.normal(0.0, 1.0, (4, par["nx"]))
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = Dop.matmat(x.T).T
    yt = Top.apply(xt)

    assert_array_equal(y, yt.detach().cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator_batch_nd(par):
    """Apply forward for input with multiple samples (= batch) and nd-arrays"""
    # temporarily, skip tests on mac as torch seems not to recognized
    # numpy when v2 is installed
    if platform.system() == "Darwin":
        return
    device = "cpu" if backend == "numpy" else "cuda"

    Dop = MatrixMult(np.random.normal(0.0, 1.0, (par["ny"], par["nx"])), otherdims=(2,))
    Top = TorchOperator(
        Dop, batch=True, flatten=False, device="cpu" if backend == "numpy" else "cuda"
    )

    x = np.random.normal(0.0, 1.0, (4, par["nx"], 2))
    xt = torch.from_numpy(to_numpy(x)).to(device)
    xt.requires_grad = True

    y = (Dop @ x.transpose(1, 2, 0)).transpose(2, 0, 1)
    yt = Top.apply(xt)

    assert_array_equal(y, yt.detach().cpu().numpy())
