import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
else:
    import numpy as np
import pytest

from pylops.basicoperators import MatrixMult
from pylops.optimization.cls_basic import CG, CGLS, LSQR
from pylops.optimization.cls_leastsquares import (
    NormalEquationsInversion,
    PreconditionedInversion,
    RegularizedInversion,
)
from pylops.optimization.cls_sparsity import FISTA, IRLS, ISTA, OMP, SPGL1, SplitBregman

# sizes are given explicitly (no np.prod) and the operators are
# set to be float64 so that nbytes = 8
N, M = 7, 5  # data and model sizes of the rectangular operator
NB = 8


def _op(n, m):
    return MatrixMult(np.random.normal(0, 1, (n, m)), dtype="float64")


def test_units(capsys):
    """Check that the estimated memory usage is returned
    in bytes and printed in the requested unit
    """
    solver = CGLS(_op(N, M))
    memuse = solver.memory_usage(show=True, unit="KB")
    assert memuse == (2 * M + 3 * N + 3 * M) * NB
    assert f"{memuse / 1024:.2f} KB" in capsys.readouterr().out


def test_cg():
    """Check the memory usage estimate of CG"""
    solver = CG(_op(M, M))
    assert solver.memory_usage() == (4 * M + 2 * M) * NB


def test_cgls():
    """Check the memory usage estimate of CGLS"""
    solver = CGLS(_op(N, M))
    assert solver.memory_usage() == (2 * M + 3 * N + 3 * M) * NB


def test_lsqr():
    """Check the memory usage estimate of LSQR"""
    solver = LSQR(_op(N, M))
    assert solver.memory_usage() == (4 * M + 2 * N + M) * NB


def test_leastsquares_regs_sum():
    """Check the memory usage estimate of LeastSquares with summed regularization sizes"""
    nopRegs = (3, 4)
    Op = _op(N, M)
    assert (
        NormalEquationsInversion(Op).memory_usage(nopRegs=nopRegs)
        == ((3 + 4) + 6 * M) * NB
    )
    assert (
        RegularizedInversion(Op).memory_usage(nopRegs=nopRegs)
        == (5 * M + 3 * (N + 3 + 4)) * NB
    )
    assert NormalEquationsInversion(Op).memory_usage() == 6 * M * NB
    assert RegularizedInversion(Op).memory_usage() == (5 * M + 3 * N) * NB


def test_preconditioned_inversion():
    """Check the memory usage estimate of PreconditionedInversion"""
    Op = _op(N, M)
    P = _op(M, 3)
    solver = PreconditionedInversion(Op)
    assert solver.memory_usage(P) == (5 * 3 + 3 * N) * NB


def test_ista_fista():
    """Check the memory usage estimate of ISTA and FISTA"""
    assert ISTA(_op(N, M)).memory_usage() == (M + N + 3 * M + N) * NB
    assert FISTA(_op(N, M)).memory_usage() == (M + N + 4 * M + N) * NB


def test_omp():
    """Check the memory usage estimate of OMP"""
    assert (
        OMP(_op(N, M)).memory_usage(niter_outer=4)
        == (2 * N + 2 * M + 5 * 4 + 2 * N) * NB
    )


@pytest.mark.parametrize(
    "kind, nel",
    [
        ("data", N + 2 * N + 5 * M + 3 * N),
        ("model", N + (N + M) + 7 * N),
        ("datamodel", (N + M) + 2 * (N + M) + 5 * M + 3 * (N + M)),
    ],
)
def test_irls(kind, nel):
    """Check the memory usage estimate of IRLS for three setups"""
    solver = IRLS(_op(N, M))
    assert solver.memory_usage(kind=kind) == nel * NB


def test_spgl1_not_implemented():
    """Check that the memory usage estimate is not available for SPGL1"""
    with pytest.raises(NotImplementedError):
        SPGL1(_op(N, M)).memory_usage()


def test_splitbregman_regs_sum():
    """Check the memory usage estimate of SplitBregman with summed regularization sizes"""
    Op = _op(N, M)
    nL1, nL2 = (3, 4), (2, 6)
    solver = SplitBregman(Op)
    expected = (M + N + 3 * (3 + 4) + (2 + 6) + (3 + 4)) * NB
    expected += (5 * M + 3 * (N + 3 + 4 + 2 + 6)) * NB
    assert solver.memory_usage(nopRegsL1=nL1, nopRegsL2=nL2) == expected
