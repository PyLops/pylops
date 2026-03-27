import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import FirstDerivative, Identity, MatrixMult
from pylops.optimization.callback import CostToInitialCallback
from pylops.optimization.cls_sparsity import IRLS
from pylops.optimization.sparsity import fista, irls, ista, omp, spgl1, splitbregman

# currently test spgl1 only if numpy<2.0.0 is installed...
np_version = npp.__version__.split(".")

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
par5 = {
    "ny": 21,
    "nx": 41,
    "imag": 0,
    "x0": True,
    "dtype": "float64",
}  # underdetermined real, non-zero initial guess
par1j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # square complex, zero initial guess
par2j = {
    "ny": 11,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # square complex, non-zero initial guess
par3j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": False,
    "dtype": "complex128",
}  # overdetermined complex, zero initial guess
par4j = {
    "ny": 31,
    "nx": 11,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # overdetermined complex, non-zero initial guess
par5j = {
    "ny": 21,
    "nx": 41,
    "imag": 1j,
    "x0": True,
    "dtype": "complex128",
}  # underdetermined complex, non-zero initial guess


def test_IRLS_unknown_kind():
    """Check error is raised if unknown kind is passed"""
    with pytest.raises(NotImplementedError, match="kind must be model"):
        _ = irls(Identity(5), np.ones(5), 10, kind="foo")


@pytest.mark.parametrize("par", [(par3), (par4), (par3j), (par4j)])
def test_IRLS_data(par):
    """Invert problem with outliers using data IRLS"""
    npp.random.seed(10)
    A = npp.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * npp.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.ones(par["nx"]) + par["imag"] * np.ones(par["nx"])
    x0 = (
        np.asarray(
            npp.random.normal(0, 1, par["nx"])
            + par["imag"] * npp.random.normal(0, 1, par["nx"])
        )
        if par["x0"]
        else None
    )
    y = Aop * x

    # add outlier
    y[par["ny"] - 2] *= 5

    # irls inversion
    for preallocate in [False, True]:
        xinv = irls(
            Aop,
            y,
            x0=x0,
            nouter=10,
            threshR=False,
            epsR=1e-2,
            epsI=0,
            tolIRLS=1e-3,
            kind="data",
            preallocate=preallocate,
        )[0]
        assert_array_almost_equal(x, xinv, decimal=2)


@pytest.mark.parametrize("par", [(par3), (par4), (par3j), (par4j)])
def test_IRLS_datamodel(par):
    """Invert problem with outliers using data-model IRLS"""
    npp.random.seed(10)
    A = npp.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * npp.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0

    x0 = (
        np.asarray(
            npp.random.normal(0, 1, par["nx"])
            + par["imag"] * npp.random.normal(0, 1, par["nx"])
        )
        if par["x0"]
        else None
    )
    y = Aop * x

    # add outlier
    y[par["ny"] - 2] *= 5

    # irls inversion
    for preallocate in [False, True]:
        xinv = irls(
            Aop,
            y,
            x0=x0,
            nouter=10,
            threshR=False,
            epsR=1e-2,
            epsI=0,
            tolIRLS=1e-3,
            kind="datamodel",
            preallocate=preallocate,
        )[0]
    assert_array_almost_equal(x, xinv, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_IRLS_model(par):
    """Invert problem with model IRLS"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    maxit = 100
    for preallocate in [False, True]:
        xinv = irls(
            Aop, y, nouter=maxit, tolIRLS=1e-3, kind="model", preallocate=preallocate
        )[0]
        assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_IRLS_model_stopping(par):
    """IRLS model testing stopping criterion rtol (here the class based
    solver is used as cost is not returned by the function based one)"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    maxit = 100
    rtol = 6e-1
    kwars_solver = dict(iter_lim=5) if backend == "numpy" else dict(niter=5)

    rcallback = CostToInitialCallback(rtol)
    irlssolve = IRLS(
        Aop,
        callbacks=[
            rcallback,
        ],
    )
    irlssolve.setup(y=y, epsI=0.1, tolIRLS=0, kind="model")
    _ = irlssolve.run(
        np.zeros(Aop.shape[1], dtype=Aop.dtype), nouter=maxit, **kwars_solver
    )
    assert irlssolve.cost[-2] / irlssolve.cost[0] >= rtol
    assert irlssolve.cost[-1] / irlssolve.cost[0] < rtol


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_MP(par):
    """Invert problem with MP"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    sigma = 1e-4
    maxit = 100
    for preallocate in [False, True]:
        xinv, _, _ = omp(
            Aop,
            y,
            maxit,
            niter_inner=0,
            optimal_coeff=True,
            sigma=sigma,
            preallocate=preallocate,
        )
        assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_OMP(par):
    """Invert problem with OMP"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    sigma = 1e-4
    maxit = 100
    for preallocate in [False, True]:
        xinv, _, _ = omp(Aop, y, maxit, sigma=sigma, preallocate=preallocate)
        assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_OMP_stopping(par):
    """OMP testing stopping criterion rtol"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    maxit = 100

    # test CostToInitialCallback callback
    for preallocate in [False, True]:
        rtol = 1e-2
        _, _, cost = omp(Aop, y, maxit, sigma=0.0, rtol=rtol, preallocate=preallocate)
        assert cost[-2] / cost[0] >= rtol
        assert cost[-1] / cost[0] < rtol

    # test CostToDataCallback callback
    for preallocate in [False, True]:
        ynorm = np.linalg.norm(y)
        rtol = 1e-2
        _, _, cost = omp(Aop, y, maxit, sigma=0.0, rtol1=rtol, preallocate=preallocate)
        assert cost[-2] / ynorm >= rtol
        assert cost[-1] / ynorm < rtol


def test_ISTA_FISTA_unknown_threshkind():
    """Check error is raised if unknown threshkind is passed"""
    with pytest.raises(ValueError, match="threshkind must be"):
        _ = ista(Identity(5), np.ones(5), 10, threshkind="foo")

    with pytest.raises(ValueError, match="threshkind must be"):
        _ = fista(Identity(5), np.ones(5), 10, threshkind="foo")


def test_ISTA_FISTA_missing_perc():
    """Check error is raised if perc=None and threshkind is percentile based"""
    with pytest.raises(ValueError, match="Provide a percentile"):
        _ = ista(Identity(5), np.ones(5), 10, perc=None, threshkind="soft-percentile")

    with pytest.raises(ValueError, match="Provide a percentile"):
        _ = fista(Identity(5), np.ones(5), 10, perc=None, threshkind="soft-percentile")


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_ISTA_FISTA_alpha_too_high(par):
    """Check error is raised or solver is stopped when alpha is chosen
    too high"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    y = Aop * x

    for solver in [ista, fista]:
        # check that exception is raised
        with pytest.raises(ValueError, match="due to residual increasing"):
            _, _, _ = solver(
                Aop,
                y,
                niter=100,
                eps=0.1,
                alpha=1e5,
                monitorres=True,
                tol=0,
            )

        # check that CostNanInfCallback catches cost=np.inf
        _, _, cost = solver(
            Aop,
            y,
            niter=100,
            eps=0.1,
            alpha=1e5,
            tol=0,
        )
        assert np.isinf(cost[-1])


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_ISTA_FISTA(par):
    """Invert problem with ISTA/FISTA"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    # Some parameters need to be tuned differently for different problem sizes
    eps = 1.0 if par["ny"] >= par["nx"] else 2.0
    perc = 50 if par["ny"] >= par["nx"] else 30
    maxit = 500

    # Regularization based ISTA and FISTA
    threshkinds = ["hard", "soft", "half"] if backend == "numpy" else ["soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            for solver in [ista, fista]:
                xinv, _, _ = solver(
                    Aop,
                    y,
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                    preallocate=preallocate,
                )
                assert_array_almost_equal(x, xinv, decimal=1)

    # Percentile based ISTA and FISTA
    if backend == "numpy":
        for threshkind in ["hard-percentile", "soft-percentile", "half-percentile"]:
            for preallocate in [False, True]:
                for solver in [ista, fista]:
                    xinv, _, _ = solver(
                        Aop,
                        y,
                        niter=maxit,
                        perc=perc,
                        threshkind=threshkind,
                        tol=0,
                        preallocate=preallocate,
                    )
                    assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_ISTA_FISTA_multiplerhs(par):
    """Invert problem with ISTA/FISTA with multiple RHS"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    x = np.outer(x, np.ones(3))
    y = Aop * x

    # Some parameters need to be tuned differently for different problem sizes
    eps = 1.0 if par["ny"] >= par["nx"] else 2.0
    perc = 50 if par["ny"] >= par["nx"] else 30
    maxit = 500

    # Regularization based ISTA and FISTA
    threshkinds = ["hard", "soft", "half"] if backend == "numpy" else ["soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            for solver in [ista, fista]:
                xinv, _, _ = solver(
                    Aop,
                    y,
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0,
                    preallocate=preallocate,
                )
                assert_array_almost_equal(x, xinv, decimal=1)

    # Percentile based ISTA and FISTA
    if backend == "numpy":
        for threshkind in ["hard-percentile", "soft-percentile", "half-percentile"]:
            for preallocate in [False, True]:
                for solver in [ista, fista]:
                    xinv, _, _ = solver(
                        Aop,
                        y,
                        niter=maxit,
                        perc=perc,
                        threshkind=threshkind,
                        tol=0,
                        preallocate=preallocate,
                    )
                    assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par3), (par5), (par1j), (par3j), (par5j)])
def test_ISTA_FISTA_stopping(par):
    """ISTA/FISTA testing stopping criterion rtol"""
    npp.random.seed(42)
    A = npp.random.randn(par["ny"], par["nx"]) + par["imag"] * npp.random.randn(
        par["ny"], par["nx"]
    )
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1.0 + par["imag"] * 1.0
    x[3] = 1.0 + par["imag"] * 1.0
    x[par["nx"] - 4] = -1.0 - par["imag"] * 1.0
    y = Aop * x

    eps = 0.5
    maxit = 500
    rtol = 5e-1

    # Regularization based ISTA and FISTA
    threshkinds = ["hard", "soft", "half"] if backend == "numpy" else ["soft", "half"]
    for threshkind in threshkinds:
        for preallocate in [False, True]:
            for solver in [ista, fista]:
                _, _, cost = solver(
                    Aop,
                    y,
                    niter=maxit,
                    eps=eps,
                    threshkind=threshkind,
                    tol=0.0,
                    rtol=rtol,
                    preallocate=preallocate,
                )
                assert cost[-2] / cost[0] >= rtol
                assert cost[-1] / cost[0] < rtol


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize(
    "par", [(par1), (par2), (par3), (par4), (par5), (par1j), (par3j)]
)
def test_SPGL1(par):
    """Invert problem with SPGL1"""
    npp.random.seed(42)
    A = npp.random.normal(0, 10, (par["ny"], par["nx"])) + par[
        "imag"
    ] * npp.random.normal(0, 10, (par["ny"], par["nx"]))
    Aop = MatrixMult(np.asarray(A), dtype=par["dtype"])

    x = np.zeros(par["nx"]) + par["imag"] * np.zeros(par["nx"])
    x[par["nx"] // 2] = 1
    x[3] = 1
    x[par["nx"] - 4] = -1

    x0 = (
        np.random.normal(0, 10, par["nx"])
        + par["imag"] * np.random.normal(0, 10, par["nx"])
        if par["x0"]
        else None
    )
    y = Aop * x
    xinv = spgl1(Aop, y, x0=x0, iter_lim=5000)[0]

    assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_SplitBregman(par):
    """Invert denoise problem with SplitBregman"""
    np.random.seed(42)
    # need enough samples for TV regularization to be effective
    nx = 3 * par["nx"]
    Iop = Identity(nx)
    Dop = FirstDerivative(nx, edge=True)

    x = np.zeros(nx)
    x[: nx // 2] = 10
    x[nx // 2 : 3 * nx // 4] = -5
    n = np.random.normal(0, 1, nx)
    y = x + n
    mu = 0.05
    lamda = 0.3
    niter_end = 50
    niter_in = 3

    x0 = np.ones(nx)
    kwars_solver = (
        dict(iter_lim=5, damp=1e-3) if backend == "numpy" else dict(niter=5, damp=1e-3)
    )

    for preallocate in [False, True]:
        xinv, _, _ = splitbregman(
            Iop,
            y,
            [Dop],
            niter_outer=niter_end,
            niter_inner=niter_in,
            mu=mu,
            epsRL1s=[lamda],
            tol=1e-4,
            tau=1,
            x0=x0 if par["x0"] else None,
            restart=False,
            preallocate=preallocate,
            **kwars_solver,
        )
        assert (np.linalg.norm(x - xinv) / np.linalg.norm(x)) < 1e-1
