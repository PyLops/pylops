import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import FFT, FFT2D

par1 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': True, 'real': False,
        'engine': 'numpy'} # nfft=nt, complex input, numpy engine
par2 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': 1024, 'real': False,
        'engine': 'numpy'} # nfft>nt, complex input, numpy engine
par3 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': True, 'real': True,
        'engine': 'numpy'}  # nfft=nt, real input, numpy engine
par4 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': 1024, 'real': True,
        'engine': 'numpy'} # nfft>nt, real input, numpy engine
par1w = {'nt': 101, 'nx': 31, 'ny': 10,
         'nfft': True, 'real': False,
         'engine': 'fftw'}  # nfft=nt, complex input, fftw engine
par2w = {'nt': 101, 'nx': 31, 'ny': 10,
         'nfft': 1024, 'real': False,
         'engine': 'fftw'}  # nfft>nt, complex input, fftw engine
par3w = {'nt': 101, 'nx': 31, 'ny': 10,
         'nfft': True, 'real': True,
         'engine': 'fftw'}  # nfft=nt, real input, fftw engine
par4w = {'nt': 101, 'nx': 31, 'ny': 10,
         'nfft': 1024, 'real': True,
         'engine': 'fftw'}  # nfft>nt, real input, fftw engine


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_1dsignal(par):
    """Dot-test and inversion for FFT operator for 1d signal
    """
    dt = 0.005
    t = np.arange(par['nt']) * dt
    f0 = 10
    x = np.sin(2 * np.pi * f0 * t)
    nfft = par['nt'] if isinstance(par['nfft'], bool) else par['nfft']
    FFTop = FFT(dims=[par['nt']], nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, par['nt'], par['nt'],
                       complexflag=0)
    else:
        assert dottest(FFTop, nfft, par['nt'],
                       complexflag=2)
        assert dottest(FFTop, nfft, par['nt'],
                       complexflag=3)

    y = FFTop * x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=8)
    assert_array_almost_equal(x, xinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal
    (fft on single dimension)
    """
    dt = 0.005
    nt, nx = par['nt'], par['nx']
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)

    # 1st dimension
    nfft = par['nt'] if isinstance(par['nfft'], bool) else par['nfft']
    FFTop = FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx, nt * nx,
                       complexflag=0)
    else:
        assert dottest(FFTop, nfft * nx, nt * nx,
                       complexflag=2)
        assert dottest(FFTop, nfft * nx, nt * nx,
                       complexflag=3)

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)

    # 2nd dimension
    nfft = par['nx'] if isinstance(par['nfft'], bool) else par['nfft']
    FFTop = FFT(dims=(nt, nx), dir=1, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx, nt * nx,
                       complexflag=0)
    else:
        assert dottest(FFTop, nt * nfft, nt * nx,
                       complexflag=2)
        assert dottest(FFTop, nt * nfft, nt * nx,
                       complexflag=3)

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_3dsignal(par):
    """Dot-test and inversion for fft operator for 3d signal
    (fft on single dimension)
    """
    dt = 0.005
    nt, nx, ny = par['nt'], par['nx'], par['ny']
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, ny])

    # 1st dimension
    nfft = par['nt'] if isinstance(par['nfft'], bool) else par['nfft']
    FFTop = FFT(dims=(nt, nx, ny), dir=0, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0)
    else:
        assert dottest(FFTop, nfft * nx * ny, nt * nx * ny,
                       complexflag=2)
        assert dottest(FFTop, nfft * nx * ny, nt * nx * ny,
                       complexflag=3)

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)

    # 2nd dimension
    if isinstance(par['nfft'], bool):
        nfft = par['nx']
    FFTop = FFT(dims=(nt, nx, ny), dir=1, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0)
    else:
        assert dottest(FFTop, nt * nfft * ny, nt * nx * ny,
                       complexflag=2)
        assert dottest(FFTop, nt * nfft * ny, nt * nx * ny,
                       complexflag=3)

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)

    # 3rd dimension
    if isinstance(par['nfft'], bool):
        nfft = par['ny']
    FFTop = FFT(dims=(nt, nx, ny), dir=2, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0)
    else:
        assert dottest(FFTop, nt * nx * nfft, nt * nx * ny,
                       complexflag=2)
        assert dottest(FFTop, nt * nx * nfft, nt * nx * ny,
                       complexflag=3)

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT2D(par):
    """Dot-test and inversion for FFT2D operator for 2d signal
    """
    dt, dx = 0.005, 5
    t = np.arange(par['nt']) * dt
    f0 = 10
    nfft1 = par['nt'] if isinstance(par['nfft'], bool) else par['nfft']
    nfft2 = par['nx'] if isinstance(par['nfft'], bool) else par['nfft']
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par['nx']) + 1)

    FFTop = FFT2D(dims=(par['nt'], par['nx']), nffts=(nfft1, nfft2),
                  sampling=(dt, dx))
    assert dottest(FFTop, nfft1*nfft2, par['nt']*par['nx'], complexflag=2)

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par['nt'], par['nx'])
    dinv = np.real(dinv).reshape(par['nt'], par['nx'])

    assert_array_almost_equal(d, dadj, decimal=8)
    assert_array_almost_equal(d, dinv, decimal=8)
