import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pylops.utils.signalprocessing import *

par1 = {'nt': 51, 'imag':0, 'dtype':'float32'} # real
par2 = {'nt': 51, 'imag':1j, 'dtype':'complex64'} # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_convmtx(par):
    """Compare convmtx with np.convolve
    """
    x = np.random.normal(0, 1, par['nt']) + \
        par['imag'] * np.random.normal(0, 1, par['nt'])

    nh = 7
    h = np.hanning(7)
    H = convmtx(h, par['nt'])
    H = H[:, nh//2:-nh//2+1]

    y = np.convolve(x, h, mode='same')
    y1 = np.dot(H, x)
    assert_array_almost_equal(y, y1, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_nonstationary_convmtx(par):
    """Compare nonstationary_convmtx with convmtx for stationary filter
    """
    x = np.random.normal(0, 1, par['nt']) + \
        par['imag'] * np.random.normal(0, 1, par['nt'])

    nh = 7
    h = np.hanning(7)
    H = convmtx(h, par['nt'])
    H = H[:, nh//2:-nh//2+1]

    H1 = \
        nonstationary_convmtx(np.repeat(h[:, np.newaxis], par['nt'], axis=1).T,
                              par['nt'], hc=nh//2, pad=(par['nt'], par['nt']))
    y = np.dot(H, x)
    y1 = np.dot(H1, x)
    assert_array_almost_equal(y, y1, decimal=4)
