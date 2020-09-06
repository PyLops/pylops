import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult
from pylops.optimization.solver import cgls

par1 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': False,
        'dtype': 'float64'}  # square real, zero initial guess
par2 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # square real, non-zero initial guess
par3 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': False,
        'dtype': 'float64'}  # overdetermined real, zero initial guess
par4 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # overdetermined real, non-zero initial guess
par1j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': False,
         'dtype': 'complex64'}  # square complex, zero initial guess
par2j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # square complex, non-zero initial guess
par3j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': False,
         'dtype': 'complex64'}  # overdetermined complex, zero initial guess
par4j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # overdetermined complex, non-zero initial guess

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1j), (par2j), (par3j), (par3j)])
def test_cgls(par):
    """CGLS with linear operator
    """
    A = np.random.normal(0, 10, (par['ny'], par['nx'])) + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx']))
    Aop = MatrixMult(A, dtype='float64')

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    if par['x0']:
        x0 = np.random.normal(0, 10, par['nx']) + \
             par['imag'] * np.random.normal(0, 10, par['nx'])
    else:
        x0 = np.zeros_like(x)

    y = Aop * x
    xinv = cgls(Aop, y, x0=x0,
                niter=par['nx'], tol=1e-10, show=False)[0]

    assert_array_almost_equal(x, xinv)