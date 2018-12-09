import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.basicoperators import FirstDerivative, SecondDerivative, Laplacian

par1 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 1., 'dy': 1., 'dx': 1.} # even with unitary sampling
par2 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # even with non-unitary sampling
par3 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 1., 'dy': 1., 'dx': 1.} # odd with unitary sampling
par4 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # odd with non-unitary sampling


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_FirstDerivative(par):
    """Dot-test and forward for FirstDerivative operator
    """
    # 1d
    D1op = FirstDerivative(par['nx'], sampling=par['dx'], dtype='float32')
    assert dottest(D1op, par['nx'], par['nx'], tol=1e-3)

    x = (par['dx']*np.arange(par['nx'])) ** 2
    yana = 2*par['dx']*np.arange(par['nx'])
    y = D1op*x
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    D1op = FirstDerivative(par['ny']*par['nx'], dims=(par['ny'], par['nx']),
                           dir=0, sampling=par['dy'], dtype='float32')
    assert dottest(D1op, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    x = np.outer((par['dy']*np.arange(par['ny']))**2, np.ones(par['nx']))
    yana = np.outer(2*par['dy']*np.arange(par['ny']), np.ones(par['nx']))
    y = D1op * x.flatten()
    y = y.reshape(par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 2d - derivative on 2nd direction
    D1op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir=1, sampling=4., dtype='float32')
    assert dottest(D1op, par['ny'] * par['nx'], par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dy'] * np.arange(par['ny'])) ** 2, np.ones(par['nx']))
    yana = np.zeros((par['ny'], par['nx']))
    y = D1op * x.flatten()
    y = y.reshape(par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 1st direction
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=0, sampling=par['dz'], dtype='float32')
    assert dottest(D1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dz']*np.arange(par['nz']))**2,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    yana = np.outer(2*par['dz']*np.arange(par['nz']),
                    np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    y = D1op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 2nd direction
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=1, sampling=par['dy'], dtype='float32')
    assert dottest(D1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    x = np.outer((par['dz'] * np.arange(par['nz'])) ** 2,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    yana = np.zeros((par['nz'], par['ny'], par['nx']))
    y = D1op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 3rd direction
    D1op = FirstDerivative(par['nz']*par['ny']*par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=2, sampling=par['dx'], dtype='float32')
    assert dottest(D1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    yana = np.zeros((par['nz'], par['ny'], par['nx']))
    y = D1op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_SecondDerivative(par):
    """Dot-test and forward for  SecondDerivative operator
    """
    # 1d
    D2op = SecondDerivative(par['nx'], sampling=par['dx'], dtype='float32')
    assert dottest(D2op, par['nx'], par['nx'], tol=1e-3)

    x = (par['dx']*np.arange(par['nx']))**3
    yana = 6*par['dx']**2*np.arange(par['nx'])
    y = D2op*x
    assert_array_almost_equal(y[2:-2], yana[2:-2], decimal=1)

    # 2d - derivative on 1st direction
    D2op = SecondDerivative(par['ny']*par['nx'],
                            dims=(par['ny'], par['nx']),
                            dir=0, sampling=par['dy'], dtype='float32')
    assert dottest(D2op, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    x = np.outer((par['dy']*np.arange(par['ny']))**3, np.ones(par['nx']))
    yana = np.outer(6*par['dy']**2*np.arange(par['ny']), np.ones(par['nx']))
    y = D2op*x.flatten()
    y = y.reshape(par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 2d - derivative on 2nd direction
    D2op = SecondDerivative(par['ny']*par['nx'],
                            dims=(par['ny'], par['nx']),
                            dir=1, sampling=par['dx'], dtype='float32')
    assert dottest(D2op, par['ny']*par['nx'],
                   par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dy']*np.arange(par['ny']))**3, np.ones(par['nx']))
    yana = np.zeros((par['ny'], par['nx']))
    y = D2op * x.flatten()
    y = y.reshape(par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 1st direction
    D2op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=0, sampling=par['dz'], dtype='float32')
    assert dottest(D2op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dz']*np.arange(par['nz']))**3,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    yana = np.outer(6*par['dz']**2*np.arange(par['nz']),
                    np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    y = D2op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 2nd direction
    D2op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=1, sampling=par['dy'], dtype='float32')
    assert dottest(D2op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dz']*np.arange(par['nz']))**3,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    yana = np.zeros((par['nz'], par['ny'], par['nx']))
    y = D2op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)

    # 3d - derivative on 3rd direction
    D2op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=2, sampling=par['dx'], dtype='float32')
    assert dottest(D2op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = np.outer((par['dz']*np.arange(par['nz']))**3,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'], par['ny'], par['nx'])
    yana = np.zeros((par['nz'], par['ny'], par['nx']))
    y = D2op * x.flatten()
    y = y.reshape(par['nz'], par['ny'], par['nx'])
    assert_array_almost_equal(y[1:-1], yana[1:-1], decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Laplacian(par):
    """Dot-test for Laplacian operator
    """
    # 2d - symmetrical
    Dlapop = Laplacian((par['ny'], par['nx']), dirs=(0, 1), weights=(1, 1),
                       sampling=((par['dy'], par['dx'])), dtype='float32')
    assert dottest(Dlapop, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    # 2d - asymmetrical
    Dlapop = Laplacian((par['ny'], par['nx']), dirs=(0, 1), weights=(1, 2),
                       sampling=((par['dy'], par['dx'])), dtype='float32')
    assert dottest(Dlapop, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian((par['nz'], par['ny'], par['nx']), dirs=(0, 1),
                       weights=(1, 1), sampling=((par['dy'], par['dx'])), dtype='float32')
    assert dottest(Dlapop, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = Laplacian((par['nz'], par['ny'], par['nx']), dirs=(0, 1),
                       weights=(1, 1), sampling=(par['dy'], par['dx']), dtype='float32')
    assert dottest(Dlapop, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)
