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
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import lsqr as sp_lsqr

from pylops.basicoperators import Diagonal, Identity, MatrixMult
from pylops.optimization.basic import cg, cgls, lsqr
from pylops.optimization.cls_basic import CG
from pylops.optimization.leastsquares import (
    normal_equations_inversion,
    regularized_inversion,
)

par1 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # square real, zero initial guess
par2 = {
    "ny": 11,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # square real, non-zero initial guess
par3 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": False,
    "dtype": "float64",
}  # overdetermined real, zero initial guess
par4 = {
    "ny": 31,
    "nx": 11,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # overdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # square complex, zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # square complex, non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex64",
}  # overdetermined complex, zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex64",
}  # overdetermined complex, non-zero initial guess


par1f = {**par1, "dtype": "float32"}
par1j128 = {**par1j, "dtype": "complex128"}

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_cg_preconditioner_shape(par):
    """Reject a preconditioner with incompatible dimensions."""
    Aop = Identity(par["nx"])
    M = MatrixMult(np.random.normal(size=(par["nx"] + 1, par["nx"])))
    with pytest.raises(ValueError, match="M must have shape"):
        cg(Aop, np.random.normal(size=par["nx"]), M=M)


@pytest.mark.parametrize("par", [(par1f), (par1), (par2), (par1j), (par2j), (par1j128)])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.parametrize("damp", [0.0, 0.8])
def test_cg_preconditioned(par, preallocate, damp):
    """Preconditioned CG solves the damped system and reports true residuals."""
    A = np.random.normal(size=(par["nx"], par["nx"])) + par["imag"] * np.random.normal(
        size=(par["nx"], par["nx"])
    )
    A = (A.conj().T @ A + np.eye(par["nx"])).astype(par["dtype"])
    Aop = MatrixMult(A, dtype=par["dtype"])
    M = Diagonal(1.0 / (A.diagonal().real + damp), dtype=par["dtype"])
    x = (
        np.random.normal(size=par["nx"])
        + par["imag"] * np.random.normal(size=par["nx"])
    ).astype(par["dtype"])
    x0 = np.random.normal(size=par["nx"]).astype(par["dtype"]) if par["x0"] else None
    y = Aop @ x + damp * x
    residuals = []
    xinv, iiter, cost = cg(
        Aop,
        y,
        x0=x0,
        niter=4 * par["nx"],
        damp=damp,
        tol=1e-12,
        M=M,
        preallocate=preallocate,
        callback=lambda value: residuals.append(
            float(np.linalg.norm(y - Aop @ value - damp * value))
        ),
    )
    assert len(residuals) == iiter
    assert_array_almost_equal(xinv, x, decimal=4)
    assert_array_almost_equal(cost[1:], np.asarray(residuals), decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_identity_preconditioner(par, preallocate):
    """An identity preconditioner may return its input without copying it."""
    A = np.random.normal(size=(par["nx"], par["nx"])) + par["imag"] * np.random.normal(
        size=(par["nx"], par["nx"])
    )
    A = (A.conj().T @ A + np.eye(par["nx"])).astype(par["dtype"])
    Aop = MatrixMult(A, dtype=par["dtype"])
    y = np.random.normal(size=par["nx"]) + par["imag"] * np.random.normal(
        size=par["nx"]
    )
    x0 = np.random.normal(size=par["nx"]).astype(par["dtype"]) if par["x0"] else None
    plain = cg(Aop, y, x0=x0, niter=4, tol=0.0, preallocate=preallocate)
    preconditioned = cg(
        Aop,
        y,
        x0=x0,
        niter=4,
        tol=0.0,
        M=Identity(par["nx"], dtype=par["dtype"]),
        preallocate=preallocate,
    )
    assert plain[1] == preconditioned[1]
    assert_array_almost_equal(plain[0], preconditioned[0], decimal=6)
    assert_array_almost_equal(plain[2], preconditioned[2], decimal=6)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_preconditioned_stopping(par, preallocate):
    """Small preconditioner scaling must not trigger an early stop."""
    y = np.random.normal(size=par["nx"])
    x0 = np.random.normal(size=par["nx"]) if par["x0"] else None
    Aop = Identity(par["nx"])
    M = 1e-12 * Aop
    xinv, iiter, cost = cg(
        Aop, y, x0=x0, niter=par["nx"], tol=1e-6, M=M, preallocate=preallocate
    )
    assert iiter == 1
    assert_array_almost_equal(xinv, y, decimal=10)
    assert_array_almost_equal(cost[0], np.linalg.norm(y if x0 is None else y - x0))


@pytest.mark.skipif(backend != "numpy", reason="SciPy CG is a NumPy reference")
@pytest.mark.parametrize("par", [(par1f), (par1), (par2), (par1j), (par2j), (par1j128)])
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_preconditioned_scipy(par, preallocate):
    """Compare a fixed number of PCG iterations with SciPy."""
    A = np.random.normal(size=(par["nx"], par["nx"])) + par["imag"] * np.random.normal(
        size=(par["nx"], par["nx"])
    )
    A = (A.conj().T @ A + np.eye(par["nx"])).astype(par["dtype"])
    Aop = MatrixMult(A, dtype=par["dtype"])
    M = Diagonal(1.0 / A.diagonal().real, dtype=par["dtype"])
    y = (
        np.random.normal(size=par["nx"])
        + par["imag"] * np.random.normal(size=par["nx"])
    ).astype(par["dtype"])
    x0 = np.random.normal(size=par["nx"]).astype(par["dtype"]) if par["x0"] else None
    expected = sp_cg(Aop, y, x0=x0, M=M, maxiter=3, atol=0.0, rtol=0.0)[0]
    result = cg(Aop, y, x0=x0, M=M, niter=3, tol=0.0, preallocate=preallocate)[0]
    assert_array_almost_equal(result, expected, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_cg_preconditioned_steps(par):
    """Manual setup and stepping use the same preconditioner as solve."""
    A = np.random.normal(size=(par["nx"], par["nx"])) + par["imag"] * np.random.normal(
        size=(par["nx"], par["nx"])
    )
    A = (A.conj().T @ A + np.eye(par["nx"])).astype(par["dtype"])
    Aop = MatrixMult(A, dtype=par["dtype"])
    M = Diagonal(1.0 / A.diagonal().real, dtype=par["dtype"])
    y = (
        np.random.normal(size=par["nx"])
        + par["imag"] * np.random.normal(size=par["nx"])
    ).astype(par["dtype"])
    solver = CG(Aop)
    x = solver.setup(y, M=M, tol=0.0)
    for _ in range(3):
        x = solver.step(x)
    expected = cg(Aop, y, M=M, niter=3, tol=0.0)[0]
    assert_array_almost_equal(x, expected, decimal=7)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cg_preconditioned_normal_equations(par):
    """Normal-equation inversion forwards the inverse preconditioner."""
    diagonal = np.exp(np.random.normal(size=par["nx"]))
    Aop = Diagonal(diagonal)
    M = Diagonal(1.0 / diagonal**2)
    x = np.random.normal(size=par["nx"])
    x0 = np.random.normal(size=par["nx"]) if par["x0"] else None
    xinv = normal_equations_inversion(
        Aop, Aop @ x, [], x0=x0, engine="pylops", M=M, niter=par["nx"], tol=1e-16
    )[0]
    assert_array_almost_equal(xinv, x, decimal=10)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg(par, preallocate):
    """CG with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    xinv = cg(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, preallocate=preallocate)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_damp(par, preallocate):
    """CG with damping solves the damped system (Op + damp*I) x = y.

    Verify equivalence between internal damping, adding ``damp*I`` to the
    operator (with ``damp=0`` in the solver), and the dense solution; and that
    ``damp=0`` reproduces the undamped result.
    """
    np.random.seed(10)

    n = par["nx"]
    A = np.random.normal(0, 1, (n, n)) + par["imag"] * np.random.normal(0, 1, (n, n))
    A = np.conj(A).T @ A + np.eye(n)  # symmetric positive-definite
    Aop = MatrixMult(A, dtype=par["dtype"])
    x = np.ones(n) + par["imag"] * np.ones(n)
    y = Aop * x
    damp = 0.8

    x_dense = np.linalg.solve(A + damp * np.eye(n), y)
    # adding damp*I to the operator and using damp=0 must match internal damping
    Aop_damped = MatrixMult(A + damp * np.eye(n), dtype=par["dtype"])

    x_int = cg(Aop, y, niter=4 * n, tol=1e-10, damp=damp, preallocate=preallocate)[0]
    x_ext = cg(Aop_damped, y, niter=4 * n, tol=1e-10, preallocate=preallocate)[0]
    assert_array_almost_equal(x_int, x_dense, decimal=5)
    assert_array_almost_equal(x_int, x_ext, decimal=5)

    # damp=0 reproduces the undamped solution
    x_undamped = cg(Aop, y, niter=4 * n, tol=1e-10, damp=0.0)[0]
    assert_array_almost_equal(x_undamped, np.linalg.solve(A, y), decimal=5)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_ndarray(par, preallocate):
    """CG with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = cg(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.shape
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_forceflat(par, preallocate):
    """CG with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = cg(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.ravel().shape
    assert_array_almost_equal(x.ravel(), xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cg_stopping(par, preallocate):
    """CG testing stopping criterion rtol"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    # test CostToInitialCallback callback
    rtol = 1e-2
    _, _, cost = cg(
        Aop, y, x0=x0, niter=par["nx"], tol=0, rtol=rtol, preallocate=preallocate
    )
    assert cost[-2] / cost[0] >= rtol
    assert cost[-1] / cost[0] < rtol

    # test CostToDataCallback callback
    ynorm = np.linalg.norm(y)
    rtol = 1e-2
    _, _, cost = cg(
        Aop, y, x0=x0, niter=par["nx"], tol=0, rtol1=rtol, preallocate=preallocate
    )
    assert cost[-2] / ynorm >= rtol
    assert cost[-1] / ynorm < rtol


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cgls(par, preallocate):
    """CGLS with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    xinv = cgls(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, preallocate=preallocate)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cgls_ndarray(par, preallocate):
    """CGLS with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = cgls(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.shape
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cgls_forceflat(par, preallocate):
    """CGLS with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = cgls(Aop, y, x0=x0, niter=2 * x.size, tol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.ravel().shape
    assert_array_almost_equal(x.ravel(), xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cgls_stopping(par, preallocate):
    """CGLS testing stopping criterion rtol"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    # test CostToInitialCallback callback
    rtol = 1e-2
    cost = cgls(
        Aop, y, x0=x0, niter=par["nx"], tol=0, rtol=rtol, preallocate=preallocate
    )[-1]
    assert cost[-2] / cost[0] >= rtol
    assert cost[-1] / cost[0] < rtol

    # test CostToDataCallback callback
    ynorm = np.linalg.norm(y)
    rtol = 1e-2
    cost = cgls(
        Aop, y, x0=x0, niter=par["nx"], tol=0, rtol1=rtol, preallocate=preallocate
    )[-1]
    assert cost[-2] / ynorm >= rtol
    assert cost[-1] / ynorm < rtol


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par4j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_cgls_damp(par, preallocate):
    """CGLS with damping against regularized_inversion with Identity"""
    np.random.seed(10)
    dtype = par["dtype"]

    A = (
        np.random.normal(0, 10, (par["ny"], par["nx"]))
        + par["imag"] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    ).astype(dtype)
    Aop = MatrixMult(A, dtype=dtype)
    Iop = Identity(par["nx"], dtype=dtype)

    x = (
        np.random.normal(0, 1, par["nx"])
        + par["imag"] * np.random.normal(0, 1, par["nx"])
    ).astype(dtype)
    if par["x0"]:
        x0 = (
            np.random.normal(0, 10, par["nx"])
            + par["imag"] * np.random.normal(0, 10, par["nx"])
        ).astype(dtype)
    else:
        x0 = None

    y = Aop * x

    damp = 50.0
    xinv = cgls(
        Aop,
        y,
        x0=x0,
        niter=par["nx"],
        damp=damp,
        tol=0,
        preallocate=preallocate,
    )[0]
    xinv_reg = regularized_inversion(
        Aop,
        y,
        [Iop],
        x0=x0,
        epsRs=[damp],
        engine="pylops",
        niter=par["nx"],
        tol=0,
    )[0]
    assert_array_almost_equal(xinv, xinv_reg, decimal=4)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_lsqr_pylops_scipy(par, preallocate):
    """Compare Pylops and scipy LSQR"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None
    y = Aop * x
    if par["x0"]:
        y_sp = y - Aop * x0
    else:
        y_sp = y.copy()

    xinv_sp = sp_lsqr(Aop, y_sp, iter_lim=par["nx"])[0]
    if par["x0"]:
        xinv_sp += x0

    xinv = lsqr(Aop, y, x0, niter=par["nx"], preallocate=preallocate)[0]

    assert_array_almost_equal(xinv, x, decimal=4)
    assert_array_almost_equal(xinv_sp, x, decimal=4)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par3), (par4)])
def test_lsqr_calc_var(par):
    """Compare PyLops and scipy LSQR variance computation for the diagonal of
    (A^H A)^-1 (issue #639)."""
    np.random.seed(10)

    A = np.random.normal(0, 1, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])
    y = Aop * (np.ones(par["nx"]))

    # niter is only a ceiling (atol/btol stop earlier); use scipy's default of
    # 2 * nx so both implementations run for the same number of iterations.
    niter = 2 * par["nx"]
    var = lsqr(Aop, y, x0=None, niter=niter, atol=1e-8, btol=1e-8)[9]
    var_sp = sp_lsqr(Aop, y, iter_lim=niter, atol=1e-8, btol=1e-8, calc_var=True)[9]

    assert not np.allclose(var, var[0])
    assert_array_almost_equal(var, var_sp, decimal=6)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_lsqr(par, preallocate):
    """LSQR with linear operator"""
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * np.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    if par["x0"]:
        x0 = np.random.normal(0, 10, par["nx"]) + par["imag"] * np.random.normal(
            0, 10, par["nx"]
        )
    else:
        x0 = None

    y = Aop * x

    xinv = lsqr(Aop, y, x0=x0, niter=par["nx"], atol=1e-5, preallocate=preallocate)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_lsqr_ndarray(par, preallocate):
    """LSQR with linear operator (and ndarray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"])
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = lsqr(Aop, y, x0=x0, niter=2 * x.size, atol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.shape
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par1j), (par2j), (par3j), (par3j)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_lsqr_forceflat(par, preallocate):
    """LSQR with linear operator (and forced 1darray as input and output)"""
    np.random.seed(10)

    dims = dimsd = (par["nx"], par["ny"])
    x = np.ones(dims) + par["imag"] * np.ones(dims)

    A = np.random.normal(0, 10, (x.size, x.size)) + par["imag"] * np.random.normal(
        0, 10, (x.size, x.size)
    )
    Aop = MatrixMult(A, dtype=par["dtype"], forceflat=True)
    Aop.dims = dims
    Aop.dimsd = dimsd

    if par["x0"]:
        x0 = np.random.normal(0, 10, dims) + par["imag"] * np.random.normal(0, 10, dims)
    else:
        x0 = None

    y = Aop * x

    xinv = lsqr(Aop, y, x0=x0, niter=2 * x.size, atol=0, preallocate=preallocate)[0]
    assert xinv.shape == x.ravel().shape
    assert_array_almost_equal(x.ravel(), xinv, decimal=4)
