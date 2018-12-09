import random
import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult, Smoothing1D
from pylops.optimization.leastsquares import NormalEquationsInversion, \
    RegularizedInversion, PreconditionedInversion

par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'x0':False, 'dtype':'complex64'} # square real with zero initial guess
par2 = {'ny': 11, 'nx': 11, 'imag': 0,
        'x0':True, 'dtype':'complex64'} # square real with non-zero initial guess
par3 = {'ny': 31, 'nx': 11, 'imag': 0,
        'x0':False, 'dtype':'complex64'} # overdetermined real with zero initial guess
par4 = {'ny': 31, 'nx': 11, 'imag': 0,
        'x0': True, 'dtype': 'complex64'}  # overdetermined real with non-zero initial guess
par1j = {'ny': 101, 'nx': 101, 'imag': 1j,
         'x0':False, 'dtype':'complex64'} # square complex with zero initial guess
par2j = {'ny': 101, 'nx': 101, 'imag': 1j,
         'x0': True, 'dtype': 'complex64'}  # square complex with non-zero initial guess
par3j = {'ny': 301, 'nx': 201, 'imag': 1j,
         'x0':False, 'dtype':'complex64'} # overdetermined complex with zero initial guess
par4j = {'ny': 301, 'nx': 201, 'imag': 1j,
         'x0': True, 'dtype': 'complex64'}  # overdetermined complex with non-zero initial guess


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1j), (par2j), (par3j), (par4j)])
def test_NormalEquationsInversion(par):
    """Solve normal equations in least squares sense
    """
    random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])

    Reg = MatrixMult(np.eye(par['nx']), dtype=par['dtype'])
    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    x0 = np.random.normal(0, 10, par['nx']) + \
         par['imag'] * np.random.normal(0, 10, par['nx']) if par['x0'] else None
    y = Gop*x
    xinv = NormalEquationsInversion(Gop, [Reg], y, epsI=0, epsRs=[0], x0=x0,
                                    returninfo=False, **dict(maxiter=200, tol=1e-10))
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1j), (par2j), (par3j), (par4j)])
def test_RegularizedInversion(par):
    """Solve regularized inversion in least squares sense
    """
    random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])
    Reg = MatrixMult(np.eye(par['nx']), dtype=par['dtype'])
    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    x0 = np.random.normal(0, 10, par['nx']) + \
         par['imag']*np.random.normal(0, 10, par['nx']) if par['x0'] else None
    y = Gop*x
    xinv = RegularizedInversion(Gop, [Reg], y, epsRs=[0], x0=x0,
                                returninfo=False,
                                **dict(damp=0, iter_lim=200, show=0))
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1j), (par2j), (par3j), (par4j)])
def test_PreconditionedInversion(par):
    """Solve regularized inversion in least squares sense
    """
    random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])

    Pre = Smoothing1D(nsmooth=5, dims=[par['nx']], dtype=par['dtype'])
    p = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    x = Pre*p
    x0 = np.random.normal(0, 1, par['nx']) + \
         par['imag'] * np.random.normal(0, 1, par['nx']) if par['x0'] else None
    y = Gop*x
    xinv = PreconditionedInversion(Gop, Pre, y, x0=x0,
                                   returninfo=False,
                                   **dict(damp=0, iter_lim=800, show=0))
    assert_array_almost_equal(x, xinv, decimal=2)
