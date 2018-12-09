import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import FFT, FFT2D

par1 = {'nt': 101, 'nx': 31, 'nfft': 101}  # nfft=nt
par2 = {'nt': 101, 'nx': 31, 'nfft': 1024} # nfft>nt


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT_1dsignal(par):
    """Dot-test and inversion for FFT operator for 1d signal
    """
    dt = 0.005
    t = np.arange(par['nt']) * dt
    f0 = 10
    x = np.sin(2 * np.pi * f0 * t)

    FFTop = FFT(dims=[par['nt']], nfft=par['nfft'], sampling=dt)
    assert dottest(FFTop, par['nfft'], par['nt'], complexflag=2)

    y = FFTop * x
    xadj = FFTop.H*y # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=8)
    assert_array_almost_equal(x, xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal (fft on single dimension)
    """
    dt = 0.005
    nt, nx = par['nt'], par['nx']
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)

    # 1st dimension
    FFTop = FFT(dims=(nt, nx), dir=0, nfft=par['nfft'], sampling=dt)
    assert dottest(FFTop, par['nfft']*par['nx'], par['nt']*par['nx'], complexflag=2)

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(par['nt'], par['nx']))
    dinv = np.real(dinv.reshape(par['nt'], par['nx']))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)

    # 2nd dimension
    FFTop = FFT(dims=(nt, nx), dir=1, nfft=par['nfft'], sampling=dt)
    assert dottest(FFTop, par['nt']*par['nfft'], par['nt']*par['nx'], complexflag=2)

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(par['nt'], par['nx']))
    dinv = np.real(dinv.reshape(par['nt'], par['nx']))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT2D(par):
    """Dot-test and inversion for FFT2D operator for 2d signal
    """
    dt, dx = 0.005, 5
    t = np.arange(par['nt']) * dt
    f0 = 10
    nfft1, nfft2 = par['nfft'], par['nfft']//2
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par['nx']) + 1)

    FFTop = FFT2D(dims=(par['nt'], par['nx']), nffts=(nfft1, nfft2), sampling=(dt, dx))
    assert dottest(FFTop, nfft1*nfft2, par['nt']*par['nx'], complexflag=2)

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par['nt'], par['nx'])
    dinv = np.real(dinv).reshape(par['nt'], par['nx'])

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)
