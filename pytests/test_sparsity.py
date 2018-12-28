import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult
from pylops.optimization.sparsity import IRLS

par1 = {'ny': 31, 'nx': 11, 'imag': 0,
        'x0':False,
        'dtype':'complex64'} # overdetermined real with zero initial guess
par2 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'complex64'} # overdetermined real with non-zero initial guess
par1j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0':False,
         'dtype':'complex64'} # overdetermined complex with zero initial guess
par2j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'} # overdetermined complex with non-zero
                               # initial guess


@pytest.mark.parametrize("par", [(par1), (par2),
                                 (par1j), (par2j)])
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
