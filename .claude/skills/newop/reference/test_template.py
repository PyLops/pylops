"""Skeleton test for a new PyLops operator.

Merge into the ``pytests/test_<subpackage>.py`` that matches the operator.
Keep the CuPy guard header identical to the one already in that file.
"""

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

from pylops.basicoperators import MyOperator  # noqa: F401  (adjust import)
from pylops.optimization.basic import lsqr
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "imag": 0, "dtype": "float64"}  # square real
par2 = {"ny": 21, "nx": 11, "imag": 0, "dtype": "float64"}  # overdetermined real
par1j = {"ny": 11, "nx": 11, "imag": 1j, "dtype": "complex128"}  # square complex
par2j = {"ny": 21, "nx": 11, "imag": 1j, "dtype": "complex128"}  # overdet. complex


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MyOperator(par):
    """Dot-test, forward and inversion for MyOperator"""
    param = np.arange(par["nx"]) + 1.0 + par["imag"] * (np.arange(par["nx"]) + 1.0)

    Op = MyOperator(param, dtype=par["dtype"])
    assert dottest(
        Op,
        par["ny"],
        par["nx"],
        rtol=1e-6 if par["dtype"] in ("float64", "complex128") else 1e-3,
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    y = Op * x
    assert_array_almost_equal(y, Op.todense() @ x, decimal=6)

    xinv = lsqr(Op, y, x0=np.zeros_like(x), niter=300, show=0)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


def test_MyOperator_raises():
    """Check input validation of MyOperator"""
    with pytest.raises(ValueError):
        MyOperator(np.ones(5), dims=(4,))
