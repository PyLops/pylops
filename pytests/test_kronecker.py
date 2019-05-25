import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import MatrixMult, Identity, \
    Kronecker, FirstDerivative

par1 = {'ny': 11, 'nx': 11,
        'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 21, 'nx': 11,
        'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11,
         'imag': 1j, 'dtype':'complex64'} # square imag
par2j = {'ny': 21, 'nx': 11,
         'imag': 1j, 'dtype':'complex64'} # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Kroneker(par):
    """Dot-test and inversion for Kronecker operator
    """
    np.random.seed(10)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = np.ones(par['nx']**2) + par['imag']*np.ones(par['nx']**2)

    Kop = Kronecker(MatrixMult(G1, dtype=par['dtype']),
                    MatrixMult(G2, dtype=par['dtype']),
                    dtype=par['dtype'])
    assert dottest(Kop, par['ny']**2, par['nx']**2,
                   complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Kop, Kop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=2)

@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Kroneker_Derivative(par):
    """Use Kronecker operator to apply the Derivative operator over one axis
    and compare with FirstDerivative(... dir=axis)
    """
    Dop = FirstDerivative(par['ny'], sampling=1,
                          edge=True, dtype='float32')
    D2op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir=0, sampling=1, edge=True,
                           dtype='float32')

    Kop = Kronecker(Dop,
                    Identity(par['nx'], dtype=par['dtype']),
                    dtype=par['dtype'])

    x = np.zeros((par['ny'], par['nx'])) + \
        par['imag']*np.zeros((par['ny'], par['nx']))
    x[par['ny']//2, par['nx']//2] = 1

    y = D2op * x.flatten()
    yk = Kop * x.flatten()
    assert_array_equal(y, yk)

