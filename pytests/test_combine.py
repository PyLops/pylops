import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import MatrixMult, VStack, HStack, BlockDiag

par1 = {'ny': 101, 'nx': 101, 'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 301, 'nx': 101, 'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 101, 'nx': 101, 'imag': 1j, 'dtype':'complex64'} # square imag
par2j = {'ny': 301, 'nx': 101, 'imag': 1j, 'dtype':'complex64'} # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_VStack(par):
    """Dot-test and inversion for VStack operator
    """
    np.random.seed(10)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])

    Vop = VStack([MatrixMult(G1, dtype=par['dtype']),
                  MatrixMult(G2, dtype=par['dtype'])],
                 dtype=par['dtype'])
    assert dottest(Vop, 2*par['ny'], par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Vop, Vop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par2), (par2j)])
def test_HStack(par):
    """Dot-test and inversion for HStack operator
    """
    np.random.seed(10)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    x = np.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])

    Hop = HStack([MatrixMult(G1, dtype=par['dtype']),
                  MatrixMult(G2, dtype=par['dtype'])],
                 dtype=par['dtype'])
    assert dottest(Hop, par['ny'], 2*par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Hop, Hop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_BlockDiag(par):
    """Dot-test and inversion for BlockDiag operator
    """
    np.random.seed(10)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = np.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])

    BDop = BlockDiag([MatrixMult(G1, dtype=par['dtype']),
                      MatrixMult(G2, dtype=par['dtype'])],
                     dtype=par['dtype'])
    assert dottest(BDop, 2*par['ny'], 2*par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(BDop, BDop * x, damp=1e-20, iter_lim=500, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)
