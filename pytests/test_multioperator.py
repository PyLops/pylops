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

from pylops.basicoperators import HStack, MatrixMult, VStack

par1 = {"ny": 11, "nx": 7, "imag": 0, "dtype": "float64"}
par1s = {"ny": 11, "nx": 7, "imag": 0, "dtype": "float32"}
par1j = {"ny": 11, "nx": 7, "imag": 1j, "dtype": "complex128"}
par1js = {"ny": 11, "nx": 7, "imag": 1j, "dtype": "complex64"}


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("Stack", [HStack, VStack])
def test_MultiOperator_parallel_kind_readonly(par, Stack):
    """Changing parallel_kind after construction raises AttributeError."""
    np.random.seed(0)
    G = (
        np.random.normal(size=(par["ny"], par["nx"]))
        + par["imag"] * np.random.normal(size=(par["ny"], par["nx"]))
    ).astype(par["dtype"])
    Op = Stack(
        [MatrixMult(G, dtype=par["dtype"])] * 2,
        parallel_kind="multithread",
        dtype=par["dtype"],
    )
    with pytest.raises(
        AttributeError, match="parallel_kind cannot be changed after construction"
    ):
        Op.parallel_kind = "multiproc"
    assert Op.parallel_kind == "multithread"


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par1js)])
@pytest.mark.parametrize("Stack", [HStack, VStack])
@pytest.mark.parametrize("initial_nproc", [1, 2])
def test_MultiOperator_thread_pool_resize(par, Stack, initial_nproc):
    """Dot-test, inversion and serial consistency after starting and resizing threads."""
    np.random.seed(0)
    G = (
        np.random.normal(size=(par["ny"], par["nx"]))
        + par["imag"] * np.random.normal(size=(par["ny"], par["nx"]))
    ).astype(par["dtype"])
    Ops = [MatrixMult(G, dtype=par["dtype"])] * 4
    Opserial = Stack(Ops, dtype=par["dtype"])
    Opthreaded = Stack(
        Ops, nproc=initial_nproc, parallel_kind="multithread", dtype=par["dtype"]
    )

    rows = par["ny"] if Stack is HStack else 4 * par["ny"]
    cols = 4 * par["nx"] if Stack is HStack else par["nx"]
    x = (
        np.random.normal(size=cols) + par["imag"] * np.random.normal(size=cols)
    ).astype(par["dtype"])
    y = (
        np.random.normal(size=rows) + par["imag"] * np.random.normal(size=rows)
    ).astype(par["dtype"])

    for nproc in (2, 1, 3):
        Opthreaded.nproc = nproc
        # forward
        assert_array_almost_equal(Opthreaded * x, Opserial * x, decimal=4)
        # adjoint
        assert_array_almost_equal(Opthreaded.H * y, Opserial.H * y, decimal=4)

    # close pool
    Opthreaded.close()
