import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import DWT, DWT2D

par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_wavelet(par):
    """Check error is raised if unknown wavelet is chosen is passed
    """
    with pytest.raises(ValueError):
        _ = DWT(dims=par['nt'], wavelet='foo')


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_1dsignal(par):
    """Dot-test and inversion for DWT operator for 1d signal
    """
    DWTop = DWT(dims=[par['nt']], dir=0, wavelet='haar', level=3)
    x = np.random.normal(0., 1., par['nt']) + \
        par['imag'] * np.random.normal(0., 1., par['nt'])

    assert dottest(DWTop, DWTop.shape[0], DWTop.shape[1],
                   complexflag=0 if par['imag'] == 0 else 3)

    y = DWTop * x
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=8)
    assert_array_almost_equal(x, xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_2dsignal(par):
    """Dot-test and inversion for DWT operator for 2d signal
    """
    for dir in [0, 1]:
        DWTop = DWT(dims=(par['nt'], par['nx']),
                    dir=dir, wavelet='haar', level=3)
        x = np.random.normal(0., 1., (par['nt'], par['nx'])) + \
            par['imag'] * np.random.normal(0., 1., (par['nt'], par['nx']))

        assert dottest(DWTop, DWTop.shape[0], DWTop.shape[1],
                       complexflag=0 if par['imag'] == 0 else 3)

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT_3dsignal(par):
    """Dot-test and inversion for DWT operator for 3d signal
    """
    for dir in [0, 1, 2]:
        DWTop = DWT(dims=(par['nt'], par['nx'], par['ny']),
                    dir=dir, wavelet='haar', level=3)
        x = np.random.normal(0., 1., (par['nt'], par['nx'], par['ny'])) + \
            par['imag'] * np.random.normal(0., 1., (par['nt'], par['nx'], par['ny']))

        assert dottest(DWTop, DWTop.shape[0], DWTop.shape[1],
                       complexflag=0 if par['imag'] == 0 else 3)

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT2D_2dsignal(par):
    """Dot-test and inversion for DWT2D operator for 2d signal
    """
    DWTop = DWT2D(dims=(par['nt'], par['nx']),
                dirs=(0, 1), wavelet='haar', level=3)
    x = np.random.normal(0., 1., (par['nt'], par['nx'])) + \
        par['imag'] * np.random.normal(0., 1., (par['nt'], par['nx']))

    assert dottest(DWTop, DWTop.shape[0], DWTop.shape[1],
                   complexflag=0 if par['imag'] == 0 else 3)

    y = DWTop * x.ravel()
    xadj = DWTop.H * y  # adjoint is same as inverse for dwt
    xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x.ravel(), xadj, decimal=8)
    assert_array_almost_equal(x.ravel(), xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DWT2D_3dsignal(par):
    """Dot-test and inversion for DWT operator for 3d signal
    """
    for dirs in [(0, 1), (0, 2), (1, 2)]:
        DWTop = DWT2D(dims=(par['nt'], par['nx'], par['ny']),
                      dirs=dirs, wavelet='haar', level=3)
        x = np.random.normal(0., 1., (par['nt'], par['nx'], par['ny'])) + \
            par['imag'] * np.random.normal(0., 1., (par['nt'], par['nx'], par['ny']))

        assert dottest(DWTop, DWTop.shape[0], DWTop.shape[1],
                       complexflag=0 if par['imag'] == 0 else 3)

        y = DWTop * x.ravel()
        xadj = DWTop.H * y  # adjoint is same as inverse for dwt
        xinv = lsqr(DWTop, y, damp=1e-10, iter_lim=10, show=0)[0]

        assert_array_almost_equal(x.ravel(), xadj, decimal=8)
        assert_array_almost_equal(x.ravel(), xinv, decimal=8)