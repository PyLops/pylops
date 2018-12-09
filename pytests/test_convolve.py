import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr
from scipy.signal import triang

from pylops.utils import dottest
from pylops.signalprocessing import Convolve1D, Convolve2D

# filters
nfilt = (5, 7)
h1 = triang(nfilt[0], sym=True)
h2 = np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True))


par1_1d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': nfilt[0]//2, 'dir':0}  # zero phase, first direction
par2_1d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': nfilt[0]//2-1, 'dir':0}  # non-zero phase, first direction
par3_1d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': nfilt[0]//2, 'dir': 1}  # zero phase, second direction
par4_1d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': nfilt[0]//2 - 1, 'dir': 1}  # non-zero phase, second direction

par1_2d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': (nfilt[0]//2, nfilt[1]//2), 'dir':0}  # zero phase, first direction
par2_2d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': (nfilt[0]//2-1, nfilt[1]//2+1), 'dir':0}  # non-zero phase, first direction
par3_2d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': (nfilt[0]//2, nfilt[1]//2), 'dir':1}  # zero phase, second direction
par4_2d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': (nfilt[0]//2-1, nfilt[1]//2+1), 'dir':1}  # non-zero phase, second direction
par5_2d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': (nfilt[0]//2, nfilt[1]//2), 'dir':2}  # zero phase, third direction
par6_2d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': (nfilt[0]//2-1, nfilt[1]//2+1), 'dir':2}  # non-zero phase, third direction


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d)])
def test_Convolve1D(par):
    """Dot-test and inversion for Convolve1D operator
    """
    np.random.seed(10)
    #1D
    if par['dir'] == 0:
        Cop = Convolve1D(par['nx'], h=h1, offset=par['offset'], dtype='float32')
        assert dottest(Cop, par['nx'], par['nx'])

        x = np.zeros((par['nx']))
        x[par['nx']//2] = 1.
        xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
        print(np.abs(x-xlsqr).max())
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 2D
    Cop = Convolve1D(par['ny'] * par['nx'], h=h1, offset=par['offset'],
                     dims=(par['ny'], par['nx']), dir=par['dir'], dtype='float32')
    assert dottest(Cop, par['ny'] * par['nx'], par['ny'] * par['nx'])

    x = np.zeros((par['ny'], par['nx']))
    x[int(par['ny']/2-3):int(par['ny']/2+3),
      int(par['nx']/2-3):int(par['nx']/2+3)] = 1.
    x = x.flatten()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
    print(np.abs(x - xlsqr).max())
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize("par", [(par1_2d), (par2_2d), (par3_2d),
                                 (par4_2d), (par5_2d), (par6_2d)])
def test_Convolve2D(par):
    """Dot-test and inversion for convolve2D operator
    """
    # 2D on 2D
    if par['dir'] == 2:
        Cop = Convolve2D(par['ny'] * par['nx'], h=h2, offset=par['offset'],
                         dims=(par['ny'], par['nx']), dtype='float32')
        assert dottest(Cop, par['ny'] * par['nx'], par['ny'] * par['nx'])

        x = np.zeros((par['ny'], par['nx']))
        x[int(par['ny'] / 2 - 3):int(par['ny'] / 2 + 3),
          int(par['nx'] / 2 - 3):int(par['nx'] / 2 + 3)] = 1.
        x = x.flatten()
        xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
        print(np.abs(x - xlsqr).max())
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2D on 3D
    Cop = Convolve2D(par['nz'] * par['ny'] * par['nx'], h=h2, offset=par['offset'],
                     dims=[par['nz'], par['ny'], par['nx']], nodir=par['dir'], dtype='float32')
    assert dottest(Cop, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'])

    x = np.zeros((par['nz'], par['ny'], par['nx']))
    x[int(par['nz'] / 2 - 3):int(par['nz'] / 2 + 3),
      int(par['ny'] / 2 - 3):int(par['ny'] / 2 + 3),
      int(par['nx'] / 2 - 3):int(par['nx'] / 2 + 3)] = 1.
    x = x.flatten()
    xlsqr = lsqr(Cop, Cop * x, damp=1e-20, iter_lim=200, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)
