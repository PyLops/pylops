import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import FFT, FFT2D, FFTND

par1 = {'nt': 41, 'nx': 31, 'ny': 10,
        'nfft': None, 'real': False, 'engine': 'numpy', 'ffthshift': False,
        'dtype':np.complex128} # nfft=nt, complex input, numpy engine
par2 = {'nt': 41, 'nx': 31, 'ny': 10,
        'nfft': 64, 'real': False, 'engine': 'numpy', 'ffthshift': False,
        'dtype':np.complex64} # nfft>nt, complex input, numpy engine
par3 = {'nt': 41, 'nx': 31, 'ny': 10,
        'nfft': None, 'real': True, 'engine': 'numpy', 'ffthshift': False,
        'dtype':np.float64} # nfft=nt, real input, numpy engine
par4 = {'nt': 41, 'nx': 31, 'ny': 10,
        'nfft': 64, 'real': True, 'engine': 'numpy', 'ffthshift': False,
        'dtype':np.float64} # nfft>nt, real input, numpy engine
par5 = {'nt': 41, 'nx': 31, 'ny': 10,
        'nfft': 64, 'real': True, 'engine': 'numpy', 'ffthshift': True,
        'dtype':np.float32} # nfft>nt, real input and fftshift, numpy engine
par1w = {'nt': 41, 'nx': 31, 'ny': 10,
         'nfft': None, 'real': False, 'engine': 'fftw', 'ffthshift': False,
         'dtype':np.complex128} # nfft=nt, complex input, fftw engine
par2w = {'nt': 41, 'nx': 31, 'ny': 10,
         'nfft': 64, 'real': False, 'engine': 'fftw', 'ffthshift': False,
         'dtype':np.complex128} # nfft>nt, complex input, fftw engine
par3w = {'nt': 41, 'nx': 31, 'ny': 10,
         'nfft': None, 'real': True, 'engine': 'fftw', 'ffthshift': False,
         'dtype':np.float64} # nfft=nt, real input, fftw engine
par4w = {'nt': 41, 'nx': 31, 'ny': 10,
         'nfft': 64, 'real': True, 'engine': 'fftw', 'ffthshift': False,
         'dtype':np.float32} # nfft>nt, real input, fftw engine

np.random.seed(5)


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_engine(par):
    """Check error is raised if unknown engine is passed
    """
    with pytest.raises(NotImplementedError):
        _ = FFT(dims=[par['nt']], nfft=par['nfft'], sampling=0.005,
                real=par['real'], engine='foo')


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_1dsignal(par):
    """Dot-test and inversion for FFT operator for 1d signal
    """
    decimal = 3 if np.real(np.ones(1, par['dtype'])).dtype == np.float32 else 8

    dt = 0.005
    t = np.arange(par['nt']) * dt
    f0 = 10
    x = np.sin(2 * np.pi * f0 * t)
    x = x.astype(par['dtype'])
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    
    FFTop = FFT(dims=[par['nt']], nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, par['nt'], par['nt'],
                       complexflag=0, tol=10**(-decimal), verb=True)
    else:
        assert dottest(FFTop, nfft, par['nt'],
                       complexflag=2, tol=10**(-decimal), verb=True)
        assert dottest(FFTop, nfft, par['nt'],
                       complexflag=3, tol=10**(-decimal), verb=True)

    y = FFTop * x
    xadj = FFTop.H * y  # adjoint is same as inverse for fft
    xinv = lsqr(FFTop, y, damp=1e-10, iter_lim=10, show=0)[0]

    assert_array_almost_equal(x, xadj, decimal=decimal)
    assert_array_almost_equal(x, xinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal
    (fft on single dimension)
    """
    decimal = 3 if np.real(np.ones(1, par['dtype'])).dtype == np.float32 else 8

    dt = 0.005
    nt, nx = par['nt'], par['nx']
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = d.astype(par['dtype'])
    
    # 1st dimension
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    FFTop = FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx, nt * nx,
                       complexflag=0, tol=10**(-decimal))
    else:
        assert dottest(FFTop, nfft * nx, nt * nx,
                       complexflag=2, tol=10**(-decimal))
        assert dottest(FFTop, nfft * nx, nt * nx,
                       complexflag=3, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # 2nd dimension
    nfft = par['nx'] if par['nfft'] is None else par['nfft']
    FFTop = FFT(dims=(nt, nx), dir=1, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx, nt * nx,
                       complexflag=0, tol=10**(-decimal))
    else:
        assert dottest(FFTop, nt * nfft, nt * nx,
                       complexflag=2, tol=10**(-decimal))
        assert dottest(FFTop, nt * nfft, nt * nx,
                       complexflag=3, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx))
    dinv = np.real(dinv.reshape(nt, nx))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5),
                                 (par1w), (par2w), (par3w), (par4w)])
def test_FFT_3dsignal(par):
    """Dot-test and inversion for fft operator for 3d signal
    (fft on single dimension)
    """
    decimal = 3 if np.real(np.ones(1, par['dtype'])).dtype == np.float32 else 8

    dt = 0.005
    nt, nx, ny = par['nt'], par['nx'], par['ny']
    t = np.arange(nt) * dt
    f0 = 10
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, ny])
    d = d.astype(par['dtype'])

    # 1st dimension
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    FFTop = FFT(dims=(nt, nx, ny), dir=0, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0, tol=10**(-decimal))
    else:
        assert dottest(FFTop, nfft * nx * ny, nt * nx * ny,
                       complexflag=2, tol=10**(-decimal))
        assert dottest(FFTop, nfft * nx * ny, nt * nx * ny,
                       complexflag=3, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is same as inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # 2nd dimension
    nfft = par['nx'] if par['nfft'] is None else par['nfft']
    FFTop = FFT(dims=(nt, nx, ny), dir=1, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0, tol=10**(-decimal))
    else:
        assert dottest(FFTop, nt * nfft * ny, nt * nx * ny,
                       complexflag=2, tol=10**(-decimal))
        assert dottest(FFTop, nt * nfft * ny, nt * nx * ny,
                       complexflag=3, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

    # 3rd dimension
    nfft = par['ny'] if par['nfft'] is None else par['nfft']
    FFTop = FFT(dims=(nt, nx, ny), dir=2, nfft=nfft, sampling=dt,
                real=par['real'], engine=par['engine'], dtype=par['dtype'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        FFTop = FFTop.H * FFTop
        assert dottest(FFTop, nt * nx * ny, nt * nx * ny,
                       complexflag=0, tol=10**(-decimal))
    else:
        assert dottest(FFTop, nt * nx * nfft, nt * nx * ny,
                       complexflag=2, tol=10**(-decimal))
        assert dottest(FFTop, nt * nx * nfft, nt * nx * ny,
                       complexflag=3, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H * D  # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=10, show=0)[0]

    dadj = np.real(dadj.reshape(nt, nx, ny))
    dinv = np.real(dinv.reshape(nt, nx, ny))

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT2D(par):
    """Dot-test and inversion for FFT2D operator for 2d signal
    """
    decimal = 3 if np.real(np.ones(1, par['dtype'])).dtype == np.float32 else 8

    dt, dx = 0.005, 5
    t = np.arange(par['nt']) * dt
    f0 = 10
    nfft1 = par['nt'] if par['nfft'] is None else par['nfft']
    nfft2 = par['nx'] if par['nfft'] is None else par['nfft']
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par['nx']) + 1)
    d = d.astype(par['dtype'])

    FFTop = FFT2D(dims=(par['nt'], par['nx']), nffts=(nfft1, nfft2),
                  sampling=(dt, dx))
    assert dottest(FFTop, nfft1*nfft2, par['nt']*par['nx'],
                   complexflag=2, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par['nt'], par['nx'])
    dinv = np.real(dinv).reshape(par['nt'], par['nx'])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_FFT3D(par):
    """Dot-test and inversion for FFTND operator for 3d signal
    """
    decimal = 3 if np.real(np.ones(1, par['dtype'])).dtype == np.float32 else 8

    dt, dx, dy = 0.005, 5, 2
    t = np.arange(par['nt']) * dt
    f0 = 10
    nfft1 = par['nt'] if par['nfft'] is None else par['nfft']
    nfft2 = par['nx'] if par['nfft'] is None else par['nfft']
    nfft3 = par['ny'] if par['nfft'] is None else par['nfft']
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(par['nx']) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, par['ny']])
    d = d.astype(par['dtype'])

    FFTop = FFTND(dims=(par['nt'], par['nx'], par['ny']),
                  nffts=(nfft1, nfft2, nfft3),
                  sampling=(dt, dx, dy))
    assert dottest(FFTop, nfft1*nfft2*nfft3,
                   par['nt']*par['nx']*par['ny'],
                   complexflag=2, tol=10**(-decimal))

    D = FFTop * d.flatten()
    dadj = FFTop.H*D # adjoint is inverse for fft
    dinv = lsqr(FFTop, D, damp=1e-10, iter_lim=100, show=0)[0]

    dadj = np.real(dadj).reshape(par['nt'], par['nx'], par['ny'])
    dinv = np.real(dinv).reshape(par['nt'], par['nx'], par['ny'])

    assert_array_almost_equal(d, dadj, decimal=decimal)
    assert_array_almost_equal(d, dinv, decimal=decimal)

