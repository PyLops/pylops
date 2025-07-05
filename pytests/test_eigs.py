import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import MatrixMult
from pylops.optimization.eigs import power_iteration
from pylops.utils.backend import to_numpy

par1 = {"n": 21, "imag": 0, "dtype": "float32"}  # square, real
par2 = {"n": 21, "imag": 1j, "dtype": "complex64"}  # square, complex


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_power_iteration(par):
    """Max eigenvalue computation with power iteration method vs. scipy methods"""
    np.random.seed(10)

    A = np.random.randn(par["n"], par["n"]) + par["imag"] * np.random.randn(
        par["n"], par["n"]
    )
    A1 = np.conj(A.T) @ A

    # non-symmetric
    Aop = MatrixMult(A)
    eig = power_iteration(Aop, niter=200, tol=0, backend=backend)[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3

    # symmetric
    A1op = MatrixMult(A1)
    eig = power_iteration(A1op, niter=200, tol=0, backend=backend)[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A1))[0]))

    assert np.abs(np.abs(eig) - eig_np) < 1e-3
