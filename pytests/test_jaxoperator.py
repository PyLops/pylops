import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pylops import JaxOperator, MatrixMult
from pylops.utils import deps

jax_message = deps.jax_import("the jax module")

if jax_message is None:
    import jax
    import jax.numpy as jnp


par1 = {"ny": 11, "nx": 11, "dtype": np.float32}  # square
par2 = {"ny": 21, "nx": 11, "dtype": np.float32}  # overdetermined

np.random.seed(0)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
def test_JaxOperator(par):
    """Apply forward and adjoint and compare with native pylops."""
    M = np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(par["dtype"])
    Mop = MatrixMult(jnp.array(M), dtype=par["dtype"])
    Jop = JaxOperator(Mop)

    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    xjnp = jnp.array(x)

    # pylops operator
    y = Mop * x
    xadj = Mop.H * y

    # jax operator
    yjnp = Jop * xjnp
    xadjnp = Jop.rmatvecad(xjnp, yjnp)

    assert_array_equal(y, np.array(yjnp))
    assert_array_equal(xadj, np.array(xadjnp))


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator_batch(par):
    """Apply forward for input with multiple samples
    (= batch) and flattened arrays"""

    M = np.random.normal(0.0, 1.0, (par["ny"], par["nx"])).astype(par["dtype"])
    Mop = MatrixMult(jnp.array(M), dtype=par["dtype"])
    Jop = JaxOperator(Mop)
    auto_batch_matvec = jax.vmap(Jop._matvec)

    x = np.random.normal(0.0, 1.0, (4, par["nx"])).astype(par["dtype"])
    xjnp = jnp.array(x)

    y = Mop.matmat(x.T).T
    yjnp = auto_batch_matvec(xjnp)

    assert_array_almost_equal(y, np.array(yjnp), decimal=5)
