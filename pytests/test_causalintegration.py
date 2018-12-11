import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.basicoperators import CausalIntegration, \
    FirstDerivative, Smoothing2D, Laplacian
from pylops.optimization.leastsquares import \
    RegularizedInversion, PreconditionedInversion

par1 = {'nt': 20, 'nx': 101, 'dt': 1., 'imag': 0,
        'dtype':'float32'}  # even samples, real, unitary step
par2 = {'nt': 21, 'nx': 101, 'dt': 1., 'imag': 0,
        'dtype': 'float32'}  # odd samples, real, unitary step
par3 = {'nt': 20, 'nx': 101, 'dt': .3, 'imag': 0,
        'dtype': 'float32'}  # even samples, real, non-unitary step
par4 = {'nt': 21, 'nx': 101, 'dt': .3,
        'imag': 0, 'dtype': 'float32'}  # odd samples, real, non-unitary step
par1j = {'nt': 20, 'nx': 101, 'dt': 1., 'imag': 1j,
         'dtype': 'complex64'}  # even samples, complex, unitary step
par2j = {'nt': 21, 'nx': 101, 'dt': 1., 'imag': 1j,
         'dtype': 'complex64'}  # odd samples, complex, unitary step
par3j = {'nt': 20, 'nx': 101, 'dt': .3, 'imag': 1j,
         'dtype': 'complex64'}  # even samples, complex, non-unitary step
par4j = {'nt': 21, 'nx': 101, 'dt': .3,
        'imag': 1j,
         'dtype': 'complex64'}  # odd samples, complex, non-unitary step


@pytest.mark.parametrize("par", [(par1), (par2),
                                 (par3), (par4),
                                 (par1j), (par2j),
                                 (par3j), (par4j)])
def test_CausalIntegration1d(par):
    """Dot-test and inversion for CausalIntegration operator for 1d signals
    """
    t = np.arange(par['nt']) * par['dt']
    x = t + par['imag']*t

    Cop = CausalIntegration(par['nt'], sampling=par['dt'],
                            halfcurrent=False, dtype=par['dtype'])
    assert dottest(Cop, par['nt'], par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    Cop = CausalIntegration(par['nt'], sampling=par['dt'],
                            halfcurrent=True, dtype=par['dtype'])
    assert dottest(Cop, par['nt'], par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)


    # numerical integration
    y = Cop * x
    # analytical integration
    yana = t ** 2 / 2. - t[0] ** 2 / 2.\
           + par['imag'] * (t ** 2 / 2. - t[0] ** 2 / 2.) + y[0]

    assert_array_almost_equal(y, yana, decimal=4)

    # numerical derivative
    Dop = FirstDerivative(par['nt'], sampling=par['dt'], dtype=par['dtype'])
    xder = Dop * y.flatten()

    # derivative by inversion
    xinv = Cop / y

    assert_array_almost_equal(x[:-1], xder[:-1], decimal=4)
    assert_array_almost_equal(x, xinv, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2),
                                 (par3), (par4),
                                 (par1j), (par2j),
                                 (par3j), (par4j)])
def test_CausalIntegration2d(par):
    """Dot-test and inversion for CausalIntegration operator for 2d signals
    """
    dt = 0.2*par['dt'] # need lower frequency in sinusoids for stability
    t = np.arange(par['nt'])*dt
    x = np.outer(np.sin(t), np.ones(par['nx'])) + \
        par['imag']*np.outer(np.sin(t), np.ones(par['nx']))

    Cop = CausalIntegration(par['nt']*par['nx'],
                            dims=(par['nt'], par['nx']),
                            sampling=dt, dir=0,
                            halfcurrent=True, dtype=par['dtype'])
    assert dottest(Cop, par['nt']*par['nx'], par['nt']*par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # numerical integration
    y = Cop * x.flatten()
    y = y.reshape(par['nt'], par['nx'])

    # analytical integration
    yana = np.outer(-np.cos(t), np.ones(par['nx'])) + np.cos(t[0]) + \
           par['imag']*(np.outer(-np.cos(t), np.ones(par['nx'])) + np.cos(t[0]))
    yana = yana.reshape(par['nt'], par['nx'])

    assert_array_almost_equal(y, yana, decimal=2)

    # numerical derivative
    Dop = FirstDerivative(par['nt']*par['nx'], dims=(par['nt'], par['nx']),
                          dir=0, sampling=dt, dtype=par['dtype'])
    xder = Dop * y.flatten()
    xder = xder.reshape(par['nt'], par['nx'])

    # derivative by inversion
    xinv = Cop / y.flatten()
    xinv = xinv.reshape(par['nt'], par['nx'])

    assert_array_almost_equal(x[:-1], xder[:-1], decimal=2)
    assert_array_almost_equal(x, xinv, decimal=2)

