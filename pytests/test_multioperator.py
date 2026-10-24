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
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 7, "imag": 0, "dtype": "float64"}
par1s = {"ny": 11, "nx": 7, "imag": 0, "dtype": "float32"}
par1j = {"ny": 11, "nx": 7, "imag": 1j, "dtype": "complex128"}
par1js = {"ny": 11, "nx": 7, "imag": 1j, "dtype": "complex64"}

np.random.seed(0)


@pytest.mark.skipif(backend == "cupy", reason="Thread pools use NumPy arrays")
@pytest.mark.parametrize("par", [(par1), (par1s), (par1j), (par1js)])
@pytest.mark.parametrize("stack", [HStack, VStack])
@pytest.mark.parametrize("initial_nproc", [1, 2])
def test_MultiOperator_thread_pool_resize(par, stack, initial_nproc):
    """Dot-test, inversion and serial consistency after starting and resizing threads."""
    np.random.seed(0)
    matrix = (
        np.random.normal(size=(par["ny"], par["nx"]))
        + par["imag"] * np.random.normal(size=(par["ny"], par["nx"]))
    ).astype(par["dtype"])
    ops = [MatrixMult(matrix, dtype=par["dtype"])] * 4
    serial = stack(ops, dtype=par["dtype"])
    threaded = stack(
        ops, nproc=initial_nproc, parallel_kind="multithread", dtype=par["dtype"]
    )
    rows = par["ny"] if stack is HStack else 4 * par["ny"]
    cols = 4 * par["nx"] if stack is HStack else par["nx"]
    x = (
        np.random.normal(size=cols) + par["imag"] * np.random.normal(size=cols)
    ).astype(par["dtype"])
    y = (
        np.random.normal(size=rows) + par["imag"] * np.random.normal(size=rows)
    ).astype(par["dtype"])

    try:
        for nproc in (2, 1, 3):
            threaded.nproc = nproc
            assert_array_almost_equal(threaded * x, serial * x, decimal=4)
            assert_array_almost_equal(threaded.H * y, serial.H * y, decimal=4)
            assert dottest(
                threaded,
                rows,
                cols,
                complexflag=0 if par["imag"] == 0 else 3,
                rtol=1e-4,
                backend=backend,
            )
        if stack is VStack:
            recovered = lsqr(
                threaded,
                serial * x,
                x0=np.zeros_like(x),
                niter=100,
                atol=1e-8,
                btol=1e-8,
            )[0]
            assert_array_almost_equal(recovered, x, decimal=3)
    finally:
        threaded.close()
