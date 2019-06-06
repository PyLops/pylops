import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.basicoperators import Diagonal

par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_1dsignal(par):
    """Dot-test and inversion for Diagonal operator for 1d signal
    """
    for ddim in (par['nx'], par['nt']):
        d = np.arange(ddim) + 1. +\
            par['imag'] * (np.arange(ddim) + 1.)

        Dop = Diagonal(d, dtype=par['dtype'])
        assert dottest(Dop, ddim, ddim,
                       complexflag=0 if par['imag'] == 0 else 3)

        x = np.ones(ddim) + par['imag']*np.ones(ddim)
        xlsqr = lsqr(Dop, Dop * x, damp=1e-20, iter_lim=300, show=0)[0]

        assert_array_almost_equal(x, xlsqr, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_2dsignal(par):
    """Dot-test and inversion for Diagonal operator for 2d signal
    """
    for idim, ddim in enumerate((par['nx'], par['nt'])):
        d = np.arange(ddim) + 1. +\
            par['imag'] * (np.arange(ddim) + 1.)

        Dop = Diagonal(d, dims=(par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        assert dottest(Dop, par['nx']*par['nt'], par['nx']*par['nt'],
                       complexflag=0 if par['imag'] == 0 else 3)

        x = np.ones((par['nx'], par['nt'])) + \
            par['imag']*np.ones((par['nx'], par['nt']))
        xlsqr = lsqr(Dop, Dop * x.ravel(), damp=1e-20, iter_lim=300, show=0)[0]

        assert_array_almost_equal(x.ravel(), xlsqr.ravel(), decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_3dsignal(par):
    """Dot-test and inversion for Diagonal operator for 3d signal
    """
    for idim, ddim in enumerate((par['ny'], par['nx'], par['nt'])):
        d = np.arange(ddim) + 1. +\
            par['imag'] * (np.arange(ddim) + 1.)

        Dop = Diagonal(d, dims=(par['ny'], par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        assert dottest(Dop, par['ny']*par['nx']*par['nt'],
                       par['ny']*par['nx']*par['nt'],
                       complexflag=0 if par['imag'] == 0 else 3)

        x = np.ones((par['ny'], par['nx'], par['nt'])) + \
            par['imag']*np.ones((par['ny'], par['nx'], par['nt']))
        xlsqr = lsqr(Dop, Dop * x.ravel(), damp=1e-20, iter_lim=300, show=0)[0]

        assert_array_almost_equal(x.ravel(), xlsqr.ravel(), decimal=4)
