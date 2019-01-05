import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import rand
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import Regression, LinearRegression, MatrixMult, \
    Diagonal, Identity, Zero, Restriction, Flip, Symmetrize

par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # square real
par2 = {'ny': 21, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # square complex
par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # overdetermined complex
par3 = {'ny': 11, 'nx': 21, 'imag': 0,
        'dtype':'float32'}  # underdetermined real


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Regression(par):
    """Dot-test, inversion and apply for Regression operator
    """
    order = 4
    t = np.arange(par['ny'], dtype=np.float32)
    LRop = Regression(t, order=order, dtype=par['dtype'])
    assert dottest(LRop, par['ny'], order+1)

    x = np.array([1., 2., 0. , 3., -1.], dtype=np.float32)
    xlsqr = lsqr(LRop, LRop*x, damp=1e-10, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    y = LRop * x
    y1 = LRop.apply(t, x)
    assert_array_almost_equal(y, y1, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_LinearRegression(par):
    """Dot-test and inversion for LinearRegression operator
    """
    t = np.arange(par['ny'], dtype=np.float32)
    LRop = LinearRegression(t, dtype=par['dtype'])
    assert dottest(LRop, par['ny'], 2)

    x = np.array([1., 2.], dtype=np.float32)
    xlsqr = lsqr(LRop, LRop*x, damp=1e-10, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    y = LRop * x
    y1 = LRop.apply(t, x)
    assert_array_almost_equal(y, y1, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult(par):
    """Dot-test and inversion for MatrixMult operator
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'],
                                 par['nx'])).astype('float32') + \
        par['imag']*np.random.normal(0, 10, (par['ny'],
                                             par['nx'])).astype('float32')
    Gop = MatrixMult(G, dtype=par['dtype'])
    assert dottest(Gop, par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag']*np.ones(par['nx'])
    xlsqr = lsqr(Gop, Gop*x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult_sparse(par):
    """Dot-test and inversion for test_MatrixMult operator using sparse
    matrix
    """
    np.random.seed(10)
    G = rand(par['ny'], par['nx'], density=0.75).astype('float32') + \
        par['imag'] * rand(par['ny'], par['nx'], density=0.75).astype('float32')

    Gop = MatrixMult(G, dtype=par['dtype'])
    assert dottest(Gop, par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 1 else 3)

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    xlsqr = lsqr(Gop, Gop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_MatrixMult_repeated(par):
    """Dot-test and inversion for test_MatrixMult operator repeated
    along another dimension
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32') + \
        par['imag'] * np.random.normal(0, 10, (par['ny'],
                                               par['nx'])).astype('float32')
    Gop = MatrixMult(G, dims=5, dtype=par['dtype'])
    assert dottest(Gop, par['ny']*5, par['nx']*5,
                   complexflag=0 if par['imag'] == 1 else 3)

    x = (np.ones((par['nx'], 5)) +
         par['imag'] * np.ones((par['nx'], 5))).flatten()
    xlsqr = lsqr(Gop, Gop*x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Identity(par):
    """Dot-test, forward and adjoint for Identity operator
    """
    Iop = Identity(par['ny'], par['nx'], dtype=par['dtype'])
    assert dottest(Iop, par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

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
    d = np.arange(par['nx']) + 1. +\
        par['imag'] * (np.arange(par['nx']) + 1.)

    Dop = Diagonal(d, dtype=par['dtype'])
    assert dottest(Dop, par['nx'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

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
    assert dottest(Rop, Nsub, par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    y = Rop * x
    x1 = Rop.H * y
    y1 = Rop.mask(x)

    assert_array_almost_equal(y, y1[iava])
    assert_array_almost_equal(x[iava], x1[iava])


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip1D(par):
    """Dot-test, forward and adjoint for Flip operator on 1d signal
    """
    x = np.arange(par['ny']) + par['imag'] * np.arange(par['ny'])

    Fop = Flip(par['ny'], dtype=par['dtype'])
    assert dottest(Fop, par['ny'], par['ny'])

    y = Fop * x
    xadj = Fop.H * y
    assert_array_equal(x, xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip2D(par):
    """Dot-test, forward and adjoint for Flip operator on 2d signal
    """
    x = {}
    x['0'] = np.outer(np.arange(par['ny']), np.ones(par['nx'])) + \
             par['imag'] * np.outer(np.arange(par['ny']), np.ones(par['nx']))
    x['1'] = np.outer(np.ones(par['ny']), np.arange(par['nx'])) + \
             par['imag'] * np.outer(np.ones(par['ny']), np.arange(par['nx']))

    for dir in [0, 1]:
        Fop = Flip(par['ny']*par['nx'], dims=(par['ny'], par['nx']),
                   dir=dir, dtype=par['dtype'])
        assert dottest(Fop, par['ny']*par['nx'], par['ny']*par['nx'])

        y = Fop * x[str(dir)].flatten()
        xadj = Fop.H * y.flatten()
        xadj = xadj.reshape(par['ny'], par['nx'])
        assert_array_equal(x[str(dir)], xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Flip3D(par):
    """Dot-test, forward and adjoint for Flip operator on 3d signal
    """
    x = {}
    x['0'] = np.outer(np.arange(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.arange(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])

    x['1'] = np.outer(np.ones(par['ny']),
                      np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])
    x['2'] = np.outer(np.ones(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx'])

    for dir in [0, 1, 2]:
        Fop = Flip(par['ny']*par['nx']*par['nx'],
                   dims=(par['ny'], par['nx'], par['nx']),
                   dir=dir, dtype=par['dtype'])
        assert dottest(Fop, par['ny']*par['nx']*par['nx'],
                       par['ny']*par['nx']*par['nx'])

        y = Fop * x[str(dir)].flatten()
        xadj = Fop.H * y.flatten()
        xadj = xadj.reshape(par['ny'], par['nx'], par['nx'])
        assert_array_equal(x[str(dir)], xadj)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize1D(par):
    """Dot-test, forward and inverse for Symmetrize operator on 1d signal
    """
    x = np.arange(par['ny']) + par['imag'] * np.arange(par['ny'])

    Sop = Symmetrize(par['ny'], dtype=par['dtype'])
    dottest(Sop, par['ny']*2-1, par['ny'])

    y = Sop * x
    xinv = Sop / y
    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize2D(par):
    """Dot-test, forward and inverse for Symmetrize operator on 2d signal
    """
    x = {}
    x['0'] = np.outer(np.arange(par['ny']), np.ones(par['nx'])) + \
             par['imag'] * np.outer(np.arange(par['ny']), np.ones(par['nx']))
    x['1'] = np.outer(np.ones(par['ny']), np.arange(par['nx'])) + \
             par['imag'] * np.outer(np.ones(par['ny']), np.arange(par['nx']))

    for dir in [0, 1]:
        Sop = Symmetrize(par['ny']*par['nx'],
                         dims=(par['ny'], par['nx']),
                         dir=dir, dtype=par['dtype'])
        y = Sop * x[str(dir)].flatten()
        assert dottest(Sop, y.size, par['ny']*par['nx'])

        xinv = Sop / y
        assert_array_almost_equal(x[str(dir)].flatten(), xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Symmetrize3D(par):
    """Dot-test, forward and adjoint for Symmetrize operator on 3d signal
    """
    x = {}
    x['0'] = np.outer(np.arange(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.arange(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])

    x['1'] = np.outer(np.ones(par['ny']),
                      np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])
    x['2'] = np.outer(np.ones(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx'])

    for dir in [0, 1, 2]:
        Sop = Symmetrize(par['ny']*par['nx']*par['nx'],
                         dims=(par['ny'], par['nx'], par['nx']),
                         dir=dir, dtype=par['dtype'])
        y = Sop * x[str(dir)].flatten()
        assert dottest(Sop, y.size, par['ny']*par['nx']*par['nx'])

        xinv = Sop / y
        assert_array_almost_equal(x[str(dir)].flatten(), xinv, decimal=3)
