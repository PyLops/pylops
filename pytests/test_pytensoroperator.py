import numpy as np
import pytensor
import pytest
from numpy.testing import assert_array_equal

from pylops import MatrixMult, PyTensorOperator

par1 = {"ny": 11, "nx": 11, "dtype": np.float32}  # square
par2 = {"ny": 21, "nx": 11, "dtype": np.float32}  # overdetermined

np.random.seed(0)
rng = np.random.default_rng()


@pytest.mark.parametrize("par", [(par1)])
def test_PyTensorOperator(par):
    """Verify output and gradient of PyTensor function obtained from a LinearOperator."""
    Dop = MatrixMult(np.random.normal(0.0, 1.0, (par["ny"], par["nx"])))
    pytensor_op = PyTensorOperator(Dop)

    # Check gradient
    inp = np.random.randn(*pytensor_op.dims)
    pytensor.gradient.verify_grad(pytensor_op, (inp,), rng=rng)

    # Check value
    x = pytensor.tensor.dvector()
    f = pytensor.function([x], pytensor_op(x))
    out = f(inp)
    assert_array_equal(out, Dop @ inp)


@pytest.mark.parametrize("par", [(par1)])
def test_PyTensorOperator_nd(par):
    """Verify output and gradient of PyTensor function obtained from a LinearOperator
    using an ND-array."""

    otherdims = rng.choice(range(1, 3), size=rng.choice(range(2, 8)))
    Dop = MatrixMult(
        np.random.normal(0.0, 1.0, (par["ny"], par["nx"])), otherdims=otherdims
    )
    pytensor_op = PyTensorOperator(Dop)

    # Check gradient
    inp = np.random.randn(*pytensor_op.dims)
    pytensor.gradient.verify_grad(pytensor_op, (inp,), rng=rng)

    # Check value
    tensor = pytensor.tensor.TensorType(dtype="float64", shape=(None,) * inp.ndim)
    x = tensor()
    f = pytensor.function([x], pytensor_op(x))
    out = f(inp)
    assert_array_equal(out, Dop @ inp)
