import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.signalprocessing import Radon2D
from pylops.optimization.sparsity import FISTA

par1 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax':1e-2,
        'centeredh': True, 'kind': 'linear',
        'interp':True, 'onthefly':True} # linear, centered, linear interp
par2 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax': 1e-2,
        'centeredh': False, 'kind': 'linear',
        'interp':False, 'onthefly':False}  # linear, uncentered, nn interp
par3 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax': 8e-3,
        'centeredh': True, 'kind': 'parabolic',
        'interp':True, 'onthefly':True}  # parabolic, centered, linear interp
par4 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax': 8e-3,
        'centeredh': False, 'kind': 'parabolic',
        'interp':False, 'onthefly':False}  # parabolic, uncentered, nn interp
par5 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax': 5e-2,
        'centeredh': True, 'kind': 'hyperbolic',
        'interp':True, 'onthefly':True}  # hyperbolic, centered, linear interp
par6 = {'nt': 11, 'nhx': 21, 'ny': 10, 'npx':21, 'pxmax': 9e-2,
        'centeredh': False, 'kind': 'hyperbolic',
        'interp':False, 'onthefly':False}  # hyperbolic, uncentered, nn interp


@pytest.mark.parametrize("par", [(par1), (par2), (par3),
                                 (par4), (par5), (par6)])
def test_Radon2D(par):
    """Dot-test and sparse inverse for Radon2D operator
    """
    dt, dh = 0.005, 1
    t = np.arange(par['nt']) * dt
    h = np.arange(par['nhx']) * dh
    px = np.linspace(0, par['pxmax'], par['npx'])
    x = np.zeros((par['npx'], par['nt']))
    x[2, par['nt']//2] = 1
    Rop = Radon2D(t, h, px, centeredh=par['centeredh'],
                  interp=par['interp'], kind=par['kind'],
                  dtype='float64')
    assert dottest(Rop, par['nhx']*par['nt'], par['npx']*par['nt'],
                       complexflag=0)
    y = Rop * x.flatten()
    y = y.reshape(par['nhx'], par['nt'])

    xinv, _, _ = FISTA(Rop, y.flatten(), 30, eps=1e0, returninfo=True)
    assert_array_almost_equal(x.flatten(), xinv, decimal=1)
