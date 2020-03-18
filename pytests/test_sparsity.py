import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult, Identity, FirstDerivative
from pylops.optimization.sparsity import IRLS, OMP, \
    ISTA, FISTA, SPGL1, SplitBregman

par1 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': False,
        'dtype': 'float64'}  # square real, zero initial guess
par2 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # square real, non-zero initial guess
par3 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0':False,
        'dtype':'float64'} # overdetermined real, zero initial guess
par4 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'} # overdetermined real, non-zero initial guess
par5 = {'ny': 21, 'nx': 41, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # underdetermined real, non-zero initial guess
par1j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': False,
         'dtype': 'complex64'}  # square complex, zero initial guess
par2j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # square complex, non-zero initial guess
par3j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0':False,
         'dtype':'complex64'} # overdetermined complex, zero initial guess
par4j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'} # overdetermined complex, non-zero initial guess
par5j = {'ny': 21, 'nx': 41, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # underdetermined complex, non-zero initial guess


@pytest.mark.parametrize("par", [(par3), (par4), (par3j), (par4j)])
def test_IRLS(par):
    """Invert problem with IRLS
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag']*np.random.normal(0, 10, (par['ny'],
                                             par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])
    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    x0 = np.random.normal(0, 10, par['nx']) + \
         par['imag']*np.random.normal(0, 10, par['nx']) if par['x0'] else None
    y = Gop*x

    # add outlier
    y[par['ny']-2] *= 5

    # normal equations with regularization
    xinv, _ = IRLS(Gop, y, 10, threshR=False, epsR=1e-2, epsI=0,
                   x0=x0, tolIRLS=1e-3, returnhistory=False)
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par3), (par5),
                                 (par1j), (par3j), (par5j)])
def test_OMP(par):
    """Invert problem with OMP
    """
    np.random.seed(42)
    Aop = MatrixMult(np.random.randn(par['ny'], par['nx']))

    x = np.zeros(par['nx'])
    x[par['nx'] // 2] = 1
    x[3] = 1
    x[par['nx'] - 4] = -1
    y = Aop * x

    sigma = 1e-4
    maxit = 100

    xinv, _, _ = OMP(Aop, y, maxit, sigma=sigma, show=False)
    assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1)])
def test_ISTA_FISTA_unknown_threshkind(par):
    """Check error is raised if unknown threshkind is passed
    """
    with pytest.raises(NotImplementedError):
        _ = ISTA(Identity(5), np.ones(5), 10, threshkind='foo')
    with pytest.raises(NotImplementedError):
        _ = FISTA(Identity(5), np.ones(5), 10, threshkind='foo')


@pytest.mark.parametrize("par", [(par1)])
def test_ISTA_FISTA_missing_perc(par):
    """Check error is raised if perc=None and threshkind is percentile based
    """
    with pytest.raises(ValueError):
        _ = ISTA(Identity(5), np.ones(5), 10, perc=None,
                 threshkind='soft-percentile')
    with pytest.raises(ValueError):
        _ = FISTA(Identity(5), np.ones(5), 10, perc=None,
                  threshkind='soft-percentile')


@pytest.mark.parametrize("par", [(par1), (par3), (par5),
                                 (par1j), (par3j), (par5j)])
def test_ISTA_FISTA(par):
    """Invert problem with ISTA/FISTA
    """
    np.random.seed(42)
    Aop = MatrixMult(np.random.randn(par['ny'], par['nx']))

    x = np.zeros(par['nx'])
    x[par['nx'] // 2] = 1
    x[3] = 1
    x[par['nx'] - 4] = -1
    y = Aop * x

    eps = 0.5
    perc = 30
    maxit = 2000

    # ISTA with too high alpha (check that exception is raised)
    with pytest.raises(ValueError):
        xinv, _, _ = ISTA(Aop, y, maxit, eps=eps, alpha=1e5, monitorres=True,
                          tol=0, returninfo=True)

    # Regularization based ISTA and FISTA
    for threshkind in ['hard', 'soft', 'half']:
        # ISTA
        xinv, _, _ = ISTA(Aop, y, maxit, eps=eps, threshkind=threshkind,
                          tol=0, returninfo=True, show=False)
        assert_array_almost_equal(x, xinv, decimal=1)

        # FISTA
        xinv, _, _ = FISTA(Aop, y, maxit, eps=eps, threshkind=threshkind,
                           tol=0, returninfo=True, show=False)
        assert_array_almost_equal(x, xinv, decimal=1)

    # Percentile based ISTA and FISTA
    for threshkind in ['hard-percentile', 'soft-percentile', 'half-percentile']:
        # ISTA
        xinv, _, _ = ISTA(Aop, y, maxit, perc=perc, threshkind=threshkind,
                          tol=0, returninfo=True, show=False)
        assert_array_almost_equal(x, xinv, decimal=1)

        # FISTA
        xinv, _, _ = FISTA(Aop, y, maxit, perc=perc, threshkind=threshkind,
                           tol=0, returninfo=True, show=False)
        assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5),
                                 (par1j), (par3j)])
def test_SPGL1(par):
    """Invert problem with SPGL1
    """
    np.random.seed(42)
    Aop = MatrixMult(np.random.randn(par['ny'], par['nx']))

    x = np.zeros(par['nx'])
    x[par['nx'] // 2] = 1
    x[3] = 1
    x[par['nx'] - 4] = -1

    x0 = np.random.normal(0, 10, par['nx']) + \
         par['imag'] * np.random.normal(0, 10, par['nx']) if \
        par['x0'] else None
    y = Aop * x
    xinv = SPGL1(Aop, y, x0=x0, iter_lim=5000)[0]

    assert_array_almost_equal(x, xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_SplitBregman(par):
    """Invert denoise problem with SplitBregman
    """
    np.random.seed(42)
    nx = 3 * par['nx']  # need enough samples for TV regularization to be effective
    Iop = Identity(nx)
    Dop = FirstDerivative(nx, edge=True)

    x = np.zeros(nx)
    x[:nx // 2] = 10
    x[nx // 2:3 * nx // 4] = -5
    n = np.random.normal(0, 1, nx)
    y = x + n

    mu = 0.01
    lamda = 0.2
    niter_end = 100
    niter_in = 3
    x0 = np.ones(nx)
    xinv, niter = SplitBregman(Iop, [Dop], y, niter_end, niter_in,
                               mu=mu, epsRL1s=[lamda],
                               tol=1e-4, tau=1,
                               x0=x0 if par['x0'] else None,
                               restart=False, show=False,
                               **dict(iter_lim=5, damp=1e-3))
    assert (np.linalg.norm(x - xinv) / np.linalg.norm(x)) < 1e-1
