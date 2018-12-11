import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import LinearRegression, MatrixMult, Diagonal, Identity, Zero, Restriction

par1 = {'ny': 101, 'nx': 101, 'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 301, 'nx': 201, 'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 101, 'nx': 101, 'imag': 1j, 'dtype':'complex64'} # square complex
par2j = {'ny': 301, 'nx': 201, 'imag': 1j, 'dtype':'complex64'} # overdetermined complex
par3 = {'ny': 101, 'nx': 201, 'imag': 0, 'dtype':'float32'}  # underdetermined real


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_LinearRegression(par):
    """Dot-test and inversion for LinearRegression operator
    """
    t = np.arange(par['ny'])
    LRop = LinearRegression(t, dtype=par['dtype'])
    assert dottest(LRop, par['ny'], 2)

    x = np.array([1., 2.])
    xlsqr = lsqr(LRop, LRop*x, damp=1e-10, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult(par):
    """Dot-test and inversion for MatrixMult operator
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag']*np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])
    assert dottest(Gop, par['ny'], par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    xlsqr = lsqr(Gop, Gop*x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult_diagonal(par):
    """Dot-test and inversion for test_MatrixMult operator repeated
    along another dimension
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    Gop = MatrixMult(G, dims=5, dtype=par['dtype'])
    assert dottest(Gop, par['ny']*5, par['nx']*5, complexflag=0 if par['imag'] == 1 else 3)

    x = (np.ones((par['nx'], 5)) + par['imag'] * np.ones((par['nx'], 5))).flatten()
    xlsqr = lsqr(Gop, Gop*x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Identity(par):
    """Dot-test, forward and adjoint for Identity operator
    """
    Iop = Identity(par['ny'], par['nx'], dtype=par['dtype'])
    assert dottest(Iop, par['ny'], par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    y = Iop*x
    x1 = Iop.H*y

    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              y[:min(par['ny'], par['nx'])], decimal=4)
    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              x1[:min(par['ny'], par['nx'])], decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Zero(par):
    """Dot-test, forward and adjoint for Zero operator
    """
    Zop = Zero(par['ny'], par['nx'], dtype=par['dtype'])
    assert dottest(Zop, par['ny'], par['nx'])

    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    y = Zop * x
    x1 = Zop.H*y

    assert_array_almost_equal(y, np.zeros(par['ny']))
    assert_array_almost_equal(x1, np.zeros(par['nx']))


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Diagonal(par):
    """Dot-test and inversion for Diagonal operator
    """
    d = np.arange(par['nx']) + 1.

    Dop = Diagonal(d, dtype=par['dtype'])
    assert dottest(Dop, par['nx'], par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    xlsqr = lsqr(Dop, Dop * x, damp=1e-20, iter_lim=300, show=0)[0]

    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Restriction(par):
    """Dot-test, forward and adjoint for Restriction operator
    """
    # subsampling locations
    np.random.seed(10)
    perc_subsampling = 0.4
    Nsub = int(np.round(par['nx'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nx']))[:Nsub])

    Rop = Restriction(par['nx'], iava, dtype=par['dtype'])
    assert dottest(Rop, Nsub, par['nx'], complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    y = Rop * x
    x1 = Rop.H * y
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])