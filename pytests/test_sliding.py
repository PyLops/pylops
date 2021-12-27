import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.signalprocessing import Sliding1D, Sliding2D, Sliding3D

par1 = {'ny': 6,  'nx': 7, 'nt': 10,
        'npy': 15, 'nwiny': 5, 'novery': 0, 'winsy': 3,
        'npx': 10, 'nwinx': 5, 'noverx': 0, 'winsx': 2,
        'tapertype': None} # no overlap, no taper
par2 = {'ny': 6, 'nx': 7, 'nt': 10,
        'npy': 15, 'nwiny': 5, 'novery': 0, 'winsy': 3,
        'npx': 10, 'nwinx': 5, 'noverx': 0, 'winsx': 2,
        'tapertype': 'hanning'} # no overlap, with taper
par3 = {'ny': 6, 'nx': 7, 'nt': 10,
        'npy': 15, 'nwiny': 7, 'novery': 3, 'winsy': 3,
        'npx': 10, 'nwinx': 4, 'noverx': 2, 'winsx': 4,
        'tapertype': None} # overlap, no taper
par4 = {'ny': 6, 'nx': 7, 'nt': 10,
        'npy': 15, 'nwiny': 7, 'novery': 3, 'winsy': 3,
        'npx': 10, 'nwinx': 4, 'noverx': 2, 'winsx': 4,
        'tapertype': 'hanning'}  # overlap, with taper


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Sliding1D(par):
    """Dot-test and inverse for Sliding1D operator
    """
    Op = MatrixMult(np.ones((par['nwiny'], par['ny'])))

    Slid = Sliding1D(Op, dim=par['ny']*par['winsy'],
                     dimd=par['npy'],
                     nwin=par['nwiny'], nover=par['novery'],
                     tapertype=par['tapertype'])
    assert dottest(Slid, par['npy'],
                   par['ny']*par['winsy'])
    x = np.ones(par['ny']*par['winsy'])
    y = Slid * x.ravel()

    xinv = LinearOperator(Slid) / y
    assert_array_almost_equal(x.ravel(), xinv)

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Sliding2D(par):
    """Dot-test and inverse for Sliding2D operator
    """
    Op = MatrixMult(np.ones((par['nwiny'] * par['nt'], par['ny'] * par['nt'])))

    Slid = Sliding2D(Op, dims=(par['ny']*par['winsy'], par['nt']),
                     dimsd=(par['npy'], par['nt']),
                     nwin=par['nwiny'], nover=par['novery'],
                     tapertype=par['tapertype'])
    assert dottest(Slid, par['npy']*par['nt'],
                   par['ny']*par['nt']*par['winsy'])
    x = np.ones((par['ny']*par['winsy'], par['nt']))
    y = Slid * x.ravel()

    xinv = LinearOperator(Slid) / y
    assert_array_almost_equal(x.ravel(), xinv)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Sliding3D(par):
    """Dot-test and inverse for Sliding3D operator
    """
    Op = MatrixMult(np.ones((par['nwiny'] * par['nwinx'] * par['nt'],
                             par['ny'] * par['nx'] * par['nt'])))

    Slid = Sliding3D(Op,
                     dims=(par['ny']*par['winsy'],
                           par['nx']*par['winsx'], par['nt']),
                     dimsd=(par['npy'], par['npx'], par['nt']),
                     nwin=(par['nwiny'], par['nwinx']),
                     nover=(par['novery'], par['noverx']),
                     nop=(par['ny'], par['nx']),
                     tapertype=par['tapertype'])
    assert dottest(Slid, par['npy']*par['npx']*par['nt'],
                   par['ny']*par['nx']*par['nt']*par['winsy']*par['winsx'])
    x = np.ones((par['ny']*par['nx']*par['winsy']*par['winsx'], par['nt']))
    y = Slid * x.ravel()

    xinv = LinearOperator(Slid) / y
    assert_array_almost_equal(x.ravel(), xinv)
