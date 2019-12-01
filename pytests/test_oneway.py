import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.oneway import PhaseShift

np.random.seed(10)

par1 = {'ny': 8, 'nx': 10, 'nt': 20,
        'dtype': 'float32'}  # even
par2 = {'ny': 9, 'nx': 11, 'nt': 21,
        'dtype': 'complex64'}  # odd


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PhaseShift_2dsignal(par):
    """Dot-test for PhaseShift of 2d signal
    """
    vel = 1500.
    zprop = 200
    freq = np.fft.rfftfreq(par['nt'], 1.)
    kx = np.fft.fftshift(np.fft.fftfreq(par['nx'], 1.))

    Pop = PhaseShift(vel, zprop, par['nt'], freq, kx,
                     dtype=par['dtype'])
    assert dottest(Pop, par['nt'] * par['nx'], par['nt'] * par['nx'])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PhaseShift_3dsignal(par):
    """Dot-test for PhaseShift of 3d signal
    """
    vel = 1500.
    zprop = 200
    freq = np.fft.rfftfreq(par['nt'], 1.)
    kx = np.fft.fftshift(np.fft.fftfreq(par['nx'], 1.))
    ky = np.fft.fftshift(np.fft.fftfreq(par['ny'], 1.))

    Pop = PhaseShift(vel, zprop, par['nt'], freq, kx, ky,
                     dtype=par['dtype'])
    assert dottest(Pop, par['nt'] * par['nx'] * par['ny'],
                   par['nt'] * par['nx'] * par['ny'])
