import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult, VStack, Diagonal, Zero

par1 = {'ny': 11, 'nx': 11,
        'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 21, 'nx': 11,
        'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11,
         'imag': 1j, 'dtype':'complex64'} # square imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j)])
def test_eigs(par):
    """Eigenvalues estimate with ARPACK and condition number
    """
    # explicit=True
    diag = np.arange(par['nx'], 0, -1) +\
           par['imag'] * np.arange(par['nx'], 0, -1)
    Op = MatrixMult(np.vstack((np.diag(diag),
                               np.zeros((par['ny'] - par['nx'], par['nx'])))))
    eigs = Op.eigs()
    assert_array_almost_equal(diag[:eigs.size], eigs, decimal=3)

    cond = Op.cond()
    assert_array_almost_equal(np.real(cond), par['nx'], decimal=3)

    #  explicit=False
    Op = Diagonal(diag, dtype=par['dtype'])
    if par['ny'] > par['nx']:
        Op = VStack([Op, Zero(par['ny'] - par['nx'], par['nx'])])
    eigs = Op.eigs()
    assert_array_almost_equal(diag[:eigs.size], eigs, decimal=3)

    cond = Op.cond()
    assert_array_almost_equal(np.real(cond), par['nx'], decimal=3)
