import pytest
import multiprocessing
import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.signalprocessing import Radon2D, Radon3D
from pylops.optimization.sparsity import FISTA

par1 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx': 21, 'npy': 17,
        'pymax': 1e-2, 'pxmax': 2e-2,
        'centeredh': True, 'kind': 'linear', 'interp': True,
        'onthefly': False,
        'engine': 'numpy'}  # linear, centered, table, linear interp, numpy
par2 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx': 21, 'npy': 17,
        'pymax': 1e-2, 'pxmax': 2e-2,
        'centeredh': False, 'kind': 'linear', 'interp': True,
        'onthefly': True,
        'engine': 'numpy'}  # linear, uncentered, fly, linear interp, numpy
par3 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax':1e-2, 'pxmax':2e-2,
        'centeredh': True, 'kind': 'linear', 'interp':True,
        'onthefly':False,
        'engine':'numba'} # linear, centered, table, linear interp, numba
par4 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax':1e-2, 'pxmax':2e-2,
        'centeredh': False, 'kind': 'linear', 'interp':False,
        'onthefly':True,
        'engine':'numba'}  # linear, uncentered, fly, linear interp, numba
par5 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax': 8e-3, 'pxmax': 7e-3,
        'centeredh': True, 'kind': 'parabolic', 'interp':False,
        'onthefly':True, 'engine':'numpy'}  # parabolic, centered, nn interp
par6 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax': 8e-3, 'pxmax': 7e-3,
        'centeredh': False, 'kind': 'parabolic', 'interp':True,
        'onthefly':False, 'engine':'numba'}  # parabolic, uncentered, linear interp
par7 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax': 9e-2, 'pxmax': 8e-2,
        'centeredh': True, 'kind': 'hyperbolic', 'interp':True,
        'onthefly':True, 'engine':'numpy'}  # hyperbolic, centered, linear interp
par8 = {'nt': 11, 'nhx': 21, 'nhy': 10, 'npx':21, 'npy':17,
        'pymax': 7e-2, 'pxmax': 8e-2,
        'centeredh': False, 'kind': 'hyperbolic', 'interp':False,
        'onthefly':False, 'engine':'numba'}  # hyperbolic, uncentered, nn interp


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_engine(par):
    """Check error is raised if unknown engine is passed
    """
    with pytest.raises(KeyError):
        _ = Radon2D(None, None, None, engine='foo')
    with pytest.raises(KeyError):
        _ = Radon3D(None, None, None, None, None, engine='foo')


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7), (par8)])
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
                  onthefly=par['onthefly'], engine=par['engine'],
                  dtype='float64')
    assert dottest(Rop, par['nhx']*par['nt'], par['npx']*par['nt'],
                   complexflag=0)
    y = Rop * x.flatten()
    y = y.reshape(par['nhx'], par['nt'])

    xinv, _, _ = FISTA(Rop, y.flatten(), 30, eps=1e0, returninfo=True)
    assert_array_almost_equal(x.flatten(), xinv, decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7), (par8)])
def test_Radon3D(par):
    """Dot-test and sparse inverse for Radon3D operator
    """
    print('threads', multiprocessing.cpu_count())
    if par['engine'] == 'numba' and \
        multiprocessing.cpu_count() >= 4: # avoid timeout in travis

        dt, dhy, dhx = 0.005, 1 , 1
        t = np.arange(par['nt']) * dt
        hy = np.arange(par['nhy']) * dhy
        hx = np.arange(par['nhx']) * dhx
        py = np.linspace(0, par['pymax'], par['npy'])
        px = np.linspace(0, par['pxmax'], par['npx'])
        x = np.zeros((par['npy'], par['npx'], par['nt']))
        x[3, 2, par['nt']//2] = 1

        Rop = Radon3D(t, hy, hx, py, px, centeredh=par['centeredh'],
                      interp=par['interp'], kind=par['kind'],
                      onthefly=par['onthefly'], engine=par['engine'],
                      dtype='float64')
        assert dottest(Rop, par['nhy']*par['nhx']*par['nt'],
                       par['npy']*par['npx']*par['nt'],
                       complexflag=0)
        if Rop.engine == 'numba' and Rop.usetable == True: # as numpy is too slow here...
            y = Rop * x.flatten()
            y = y.reshape(par['nhy'], par['nhx'], par['nt'])

            xinv, _, _ = FISTA(Rop, y.flatten(), 200, eps=3e0, returninfo=True)
            assert_array_almost_equal(x.flatten(), xinv, decimal=1)