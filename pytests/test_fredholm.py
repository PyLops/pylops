import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.signalprocessing import Fredholm1

par1 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt':True, 'imag': 0, 'dtype': 'float32'}  # real, saved Gt
par2 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt': False, 'imag': 0, 'dtype': 'float32'}  # real, unsaved Gt
par3 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt':True, 'imag': 1j, 'dtype': 'complex64'}  # complex, saved Gt
par4 = {'nsl': 3, 'ny': 6, 'nx': 4, 'nz': 5, 'saveGt': False,
        'imag': 1j, 'dtype': 'complex64'}  # complex, unsaved Gt

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Fredholm1(par):
    """Dot-test and inversion for Fredholm1 operator
    """
    F = np.arange(par['nsl']*par['nx']*par['ny']).reshape(par['nsl'],
                                                          par['nx'],
                                                          par['ny'])
    x = np.ones((par['nsl'], par['ny'], par['nz'])) + \
        par['imag'] * np.ones((par['nsl'], par['ny'], par['nz']))

    Fop = Fredholm1(F, nz=par['nz'], saveGt=par['saveGt'], dtype=par['dtype'])
    assert dottest(Fop, par['nsl']*par['nx']*par['nz'],
                   par['nsl']*par['ny']*par['nz'],
                   complexflag=0 if par['imag'] == 0 else 3)
    xlsqr = lsqr(Fop, Fop * x.flatten(), damp=1e-20,
                 iter_lim=30, show=0)[0]
    xlsqr = xlsqr.reshape(par['nsl'], par['ny'], par['nz'])
    assert_array_almost_equal(x, xlsqr, decimal=3)
