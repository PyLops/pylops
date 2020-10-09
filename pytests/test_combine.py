import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import random as sp_random
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import MatrixMult, VStack, \
    HStack, Block, BlockDiag

par1 = {'ny': 101, 'nx': 101,
        'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 301, 'nx': 101,
        'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 101, 'nx': 101,
         'imag': 1j, 'dtype':'complex64'} # square imag
par2j = {'ny': 301, 'nx': 101,
         'imag': 1j, 'dtype':'complex64'} # overdetermined imag


@pytest.mark.parametrize("par", [(par1)])
def test_VStack_incosistent_columns(par):
    """Check error is raised if operators with different number of columns
    are passed to VStack
    """
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'] + 1)).astype(par['dtype'])
    with pytest.raises(ValueError):
        VStack([MatrixMult(G1, dtype=par['dtype']),
                MatrixMult(G2, dtype=par['dtype'])],
               dtype=par['dtype'])


@pytest.mark.parametrize("par", [(par1)])
def test_HStack_incosistent_columns(par):
    """Check error is raised if operators with different number of rows
    are passed to VStack
    """
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'] + 1, par['nx'])).astype(par['dtype'])
    with pytest.raises(ValueError):
        HStack([MatrixMult(G1, dtype=par['dtype']),
                MatrixMult(G2, dtype=par['dtype'])],
               dtype=par['dtype'])


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_VStack(par):
    """Dot-test and inversion for VStack operator
    """
    np.random.seed(0)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])

    Vop = VStack([MatrixMult(G1, dtype=par['dtype']),
                  MatrixMult(G2, dtype=par['dtype'])],
                 dtype=par['dtype'])
    assert dottest(Vop, 2*par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Vop, Vop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)

    # use numpy matrix directly in the definition of the operator
    V1op = VStack([G1, MatrixMult(G2, dtype=par['dtype'])],
                  dtype=par['dtype'])
    assert dottest(V1op, 2 * par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # use scipy matrix directly in the definition of the operator
    G1 = sp_random(par['ny'], par['nx'], density=0.4).astype('float32')
    V2op = VStack([G1, MatrixMult(G2, dtype=par['dtype'])],
                  dtype=par['dtype'])
    assert dottest(V2op, 2 * par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)


@pytest.mark.parametrize("par", [(par2), (par2j)])
def test_HStack(par):
    """Dot-test and inversion for HStack operator with numpy array as input
    """
    np.random.seed(0)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    x = np.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])

    Hop = HStack([G1, MatrixMult(G2, dtype=par['dtype'])],
                 dtype=par['dtype'])
    assert dottest(Hop, par['ny'], 2*par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Hop, Hop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)

    # use numpy matrix directly in the definition of the operator
    H1op = HStack([G1, MatrixMult(G2, dtype=par['dtype'])],
                  dtype=par['dtype'])
    assert dottest(H1op, par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # use scipy matrix directly in the definition of the operator
    G1 = sp_random(par['ny'], par['nx'], density=0.4).astype('float32')
    H2op = HStack([G1, MatrixMult(G2, dtype=par['dtype'])],
                  dtype=par['dtype'])
    assert dottest(H2op, par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Block(par):
    """Dot-test and inversion for Block operator
    """
    np.random.seed(0)
    G11 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G12 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G21 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G22 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])

    x = np.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])

    Bop = Block([[MatrixMult(G11, dtype=par['dtype']),
                  MatrixMult(G12, dtype=par['dtype'])],
                 [MatrixMult(G21, dtype=par['dtype']),
                  MatrixMult(G22, dtype=par['dtype'])]],
                dtype=par['dtype'])
    assert dottest(Bop, 2*par['ny'], 2*par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(Bop, Bop * x, damp=1e-20, iter_lim=500, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # use numpy matrix directly in the definition of the operator
    B1op = Block([[G11,
                   MatrixMult(G12, dtype=par['dtype'])],
                  [MatrixMult(G21, dtype=par['dtype']),
                   G22]], dtype=par['dtype'])
    assert dottest(B1op, 2 * par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # use scipy matrix directly in the definition of the operator
    G11 = sp_random(par['ny'], par['nx'], density=0.4).astype('float32')
    B2op = Block([[G11,
                   MatrixMult(G12, dtype=par['dtype'])],
                  [MatrixMult(G21, dtype=par['dtype']),
                   G22]], dtype=par['dtype'])
    assert dottest(B2op, 2 * par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_BlockDiag(par):
    """Dot-test and inversion for BlockDiag operator
    """
    np.random.seed(0)
    G1 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = np.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = np.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])

    BDop = BlockDiag([MatrixMult(G1, dtype=par['dtype']),
                      MatrixMult(G2, dtype=par['dtype'])],
                     dtype=par['dtype'])
    assert dottest(BDop, 2*par['ny'], 2*par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    xlsqr = lsqr(BDop, BDop * x, damp=1e-20, iter_lim=500, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # use numpy matrix directly in the definition of the operator
    BD1op = BlockDiag([MatrixMult(G1, dtype=par['dtype']), G2],
                      dtype=par['dtype'])
    assert dottest(BD1op, 2 * par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # use scipy matrix directly in the definition of the operator
    G2 = sp_random(par['ny'], par['nx'], density=0.4).astype('float32')
    BD2op = BlockDiag([MatrixMult(G1, dtype=par['dtype']), G2],
                      dtype=par['dtype'])
    assert dottest(BD2op, 2 * par['ny'], 2 * par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)
