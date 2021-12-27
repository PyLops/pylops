import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import Fredholm1

par1 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'usematmul': True, 'saveGt':True,
        'imag': 0, 'dtype': 'float32'}  # real, saved Gt
par2 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'usematmul': True, 'saveGt': False,
        'imag': 0, 'dtype': 'float32'}  # real, unsaved Gt
par3 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'usematmul': False, 'saveGt':True,
        'imag': 1j, 'dtype': 'complex64'}  # complex, saved Gt
par4 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5, 'saveGt': False,
        'usematmul': False, 'saveGt':False,
        'imag': 1j, 'dtype': 'complex64'}  # complex, unsaved Gt
par5 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 1,
        'usematmul': True, 'saveGt': True,
        'imag': 0, 'dtype': 'float32'}  # real, saved Gt, nz=1
par6 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 1,
        'usematmul': True, 'saveGt': False,
        'imag': 0, 'dtype': 'float32'}  # real, unsaved Gt, nz=1


@pytest.mark.parametrize("par", [(par1), (par2),
                                 (par3), (par4),
                                 (par5), (par6)])
def test_Fredholm1(par):
    """Dot-test and inversion for Fredholm1 operator
    """
    np.random.seed(10)

    _F = np.arange(par['nsl'] * par['nx'] * par['ny']).reshape(par['nsl'],
                                                               par['nx'],
                                                               par['ny'])
    F = _F - par['imag'] * _F

    x = np.ones((par['nsl'], par['ny'], par['nz'])) + \
        par['imag'] * np.ones((par['nsl'], par['ny'], par['nz']))

    Fop = Fredholm1(F, nz=par['nz'], saveGt=par['saveGt'],
                    usematmul=par['usematmul'], dtype=par['dtype'])
    assert dottest(Fop, par['nsl']*par['nx']*par['nz'],
                   par['nsl']*par['ny']*par['nz'],
                   complexflag=0 if par['imag'] == 0 else 3)
    xlsqr = lsqr(Fop, Fop * x.ravel(), damp=1e-20,
                 iter_lim=30, show=0)[0]
    xlsqr = xlsqr.reshape(par['nsl'], par['ny'], par['nz'])
    assert_array_almost_equal(x, xlsqr, decimal=3)
