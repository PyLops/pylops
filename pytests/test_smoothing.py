import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import Smoothing1D, Smoothing2D

par1 = {'nz': 10, 'ny': 30, 'nx': 20, 'dir':0} # even, first direction
par2 = {'nz': 11, 'ny': 51, 'nx': 31, 'dir':0} # odd, first direction
par3 = {'nz': 10, 'ny': 30, 'nx': 20, 'dir':1} # even, second direction
par4 = {'nz': 11, 'ny': 51, 'nx': 31, 'dir':1} # odd, second direction
par5 = {'nz': 10, 'ny': 30, 'nx': 20, 'dir':2} # even, fithirdrst direction
par6 = {'nz': 11, 'ny': 51, 'nx': 31, 'dir':2} # odd, third direction


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Smoothing1D(par):
    """Dot-test and inversion for smoothing
    """
    # 1d kernel on 1d signal
    D1op = Smoothing1D(nsmooth=5, dims=par['nx'], dtype='float32')
    assert dottest(D1op, par['nx'], par['nx'])

    x = np.random.normal(0, 1, par['nx'])
    y = D1op * x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # 1d kernel on 2d signal
    D1op = Smoothing1D(nsmooth=5, dims=(par['ny'], par['nx']), dir=par['dir'], dtype='float32')
    assert dottest(D1op, par['ny']*par['nx'], par['ny']*par['nx'])

    x = np.random.normal(0, 1, (par['ny'], par['nx'])).flatten()
    y = D1op*x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)

    # 1d kernel on 3d signal
    D1op = Smoothing1D(nsmooth=5, dims=(par['nz'], par['ny'], par['nx']),
                       dir=par['dir'], dtype='float32')
    assert dottest(D1op, par['nz'] * par['ny'] * par['nx'], par['nz'] * par['ny'] * par['nx'])

    x = np.random.normal(0, 1, (par['nz'], par['ny'], par['nx'])).flatten()
    y = D1op*x
    xlsqr = lsqr(D1op, y, damp=1e-10, iter_lim=100, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5), (par6)])
def test_Smoothing2D(par):
    """Dot-test for smoothing
    """
    # 2d kernel on 2d signal
    if par['dir'] < 2:
        D2op = Smoothing2D(nsmooth=(5, 5), dims=(par['ny'], par['nx']), dtype='float32')
        assert dottest(D2op, par['ny']*par['nx'], par['ny']*par['nx'])

        x = np.zeros((par['ny'], par['nx']))
        x[par['ny']//2, par['nx']//2] = 1.
        x = x.flatten()
        y = D2op*x
        xlsqr = lsqr(D2op, y, damp=1e-10, iter_lim=400, show=0)[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2d kernel on 3d signal
    D2op = Smoothing2D(nsmooth=(5, 5), dims=(par['nz'], par['ny'], par['nx']),
                       nodir=par['dir'], dtype='float32')
    assert dottest(D2op, par['nz'] * par['ny'] * par['nx'], par['nz'] * par['ny'] * par['nx'])

    x = np.zeros((par['nz'], par['ny'], par['nx']))
    x[par['nz']//2, par['ny']//2, par['nx']//2] = 1.
    x = x.flatten()
    y = D2op*x
    xlsqr = lsqr(D2op, y, damp=1e-10, iter_lim=400, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)
