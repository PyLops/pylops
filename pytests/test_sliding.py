import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.signalprocessing import Sliding2D

par1 = {'nx': 15, 'ny': 6, 'nt': 10, 'nwin': 5, 'nover': 0, 'wins': 3,
        'tapertype': None} # no overlap, no taper
par2 = {'nx': 15, 'ny': 6, 'nt': 10, 'nwin': 5, 'nover': 0, 'wins': 3,
        'tapertype': 'hanning'} # no overlap, with taper
par3 = {'nx': 15, 'ny': 6, 'nt': 10, 'nwin': 7, 'nover': 3, 'wins': 3,
        'tapertype': None} # overlap, no taper
par4 = {'nx': 15, 'ny': 6, 'nt': 10, 'nwin': 7, 'nover': 3, 'wins': 3,
        'tapertype': 'hanning'}  # overlap, with taper


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Sliding2D(par):
    """Dot-test and inverse for Sliding2D operator
    """
    Op = MatrixMult(np.ones((par['nwin'] * par['nt'], par['ny'] * par['nt'])))

    Slid = Sliding2D(Op, (par['ny']*par['wins'], par['nt']),
                     (par['nx'], par['nt']),
                     par['nwin'], par['nover'],
                     tapertype=par['tapertype'])
    assert dottest(Slid, par['nx']*par['nt'],
                   par['ny']*par['nt']*par['wins'])
    x = np.ones((par['ny']*par['wins'], par['nt']))
    y = Slid * x.flatten()

    xinv = LinearOperator(Slid) / y
    assert_array_almost_equal(x.flatten(), xinv)
