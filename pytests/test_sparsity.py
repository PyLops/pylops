import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult
from pylops.optimization.sparsity import IRLS, ISTA, FISTA

par1 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': False,
        'dtype': 'float64'}  # square real, zero initial guess
par2 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # square real, non-zero initial guess
par3 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0':False,
        'dtype':'float64'} # overdetermined real, zero initial guess
par4 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'} # overdetermined real, non-zero initial guess
par5 = {'ny': 11, 'nx': 41, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # underdetermined real, non-zero initial guess
par1j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': False,
         'dtype': 'complex64'}  # square complex, zero initial guess
par2j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # square complex, non-zero initial guess
par3j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0':False,
         'dtype':'complex64'} # overdetermined complex, zero initial guess
par4j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'} # overdetermined complex, non-zero initial guess
par5j = {'ny': 11, 'nx': 41, 'imag': 1j, 'x0': True,
        'dtype': 'complex64'}  # underdetermined complex, non-zero initial guess


@pytest.mark.parametrize("par", [(par3), (par4), (par3j), (par4j)])
def test_IRLS(par):
    """Invert problem with IRLS
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag']*np.random.normal(0, 10,
                                     (par['ny'], par['nx'])).astype('float32')
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
    maxit = 5000

    # ISTA with too high alpha (check that exception is raised)
    with pytest.raises(ValueError):
        xinv, _, _ = ISTA(Aop, y, maxit, eps=eps, alpha=1e5, monitorres=True,
                          tol=0, returninfo=True)

    # ISTA
    xinv, _, _ = ISTA(Aop, y, maxit, eps=eps,
                      tol=0, returninfo=True)
    assert_array_almost_equal(x, xinv, decimal=1)

    # FISTA
    xinv, _, _ = FISTA(Aop, y, maxit, eps=eps,
                       tol=0, returninfo=True)
    assert_array_almost_equal(x, xinv, decimal=1)
