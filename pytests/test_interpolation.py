import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.signalprocessing import Interp

par1 = {'ny': 21, 'nx': 11, 'nt':20, 'imag': 0,
        'dtype':'float32', 'kind': 'nearest'}  # real, nearest
par2 = {'ny': 21, 'nx': 11, 'nt':20, 'imag': 1j,
        'dtype':'complex64', 'kind': 'nearest'} # complex, nearest
par3 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32', 'kind': 'linear'}  # real, linear
par4 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64', 'kind': 'linear'}  # complex, linear

# subsampling factor
perc_subsampling = 0.4
np.random.seed(1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Interp_1dsignal(par):
    """Dot-test, forward and adjoint for Interp operator for 1d signal
    """
    Nsub = int(np.round(par['nx'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nx']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['nx'], iava, kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, Nsub, par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['nx'], iava + 0.3, kind=par['kind'],
                       dtype=par['dtype'])
    assert dottest(Iop, Nsub, par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # repeated indeces
    with pytest.raises(ValueError):
        iava[-2] = 0
        iava[-1] = 0
        _, _ = Interp(par['nx'], iava + 0.3,
                      kind=par['kind'], dtype=par['dtype'])

    # forward
    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    y = Iop * x
    ydec = Idecop * x

    assert_array_almost_equal(y, x[iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[iava])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Interp_2dsignal(par):
    """Dot-test, forward and adjoint for Restriction operator for 2d signal
    """
    x = np.ones((par['nx'], par['nt'])) + \
        par['imag'] * np.ones((par['nx'], par['nt']))

    # 1st direction
    Nsub = int(np.round(par['nx'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nx']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['nx']*par['nt'], iava,
                    dims=(par['nx'], par['nt']), dir=0,
                    kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, Nsub*par['nt'], par['nx']*par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['nx'] * par['nt'], iava + 0.3,
                       dims=(par['nx'], par['nt']), dir=0,
                       kind=par['kind'], dtype=par['dtype'])

    # repeated indeces
    with pytest.raises(ValueError):
        iava[-2] = 0
        iava[-1] = 0
        _, _ = Interp(par['nx'] * par['nt'], iava + 0.3,
                      dims=(par['nx'], par['nt']), dir=0,
                      kind=par['kind'], dtype=par['dtype'])

    y = (Iop * x.ravel()).reshape(Nsub, par['nt'])
    ydec = (Idecop * x.ravel()).reshape(Nsub, par['nt'])

    assert_array_almost_equal(y, x[iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[iava])


    # 2nd direction
    Nsub = int(np.round(par['nt'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nt']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['nx'] * par['nt'], iava,
                    dims=(par['nx'], par['nt']), dir=1,
                    kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, par['nx'] * Nsub, par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['nx'] * par['nt'], iava + 0.3,
                       dims=(par['nx'], par['nt']), dir=1,
                       kind=par['kind'], dtype=par['dtype'])
    assert dottest(Idecop, par['nx'] * Nsub, par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    y = (Iop * x.ravel()).reshape(par['nx'], Nsub)
    ydec = (Idecop * x.ravel()).reshape(par['nx'], Nsub)

    assert_array_almost_equal(y, x[:, iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[:, iava])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Interp_3dsignal(par):
    """Dot-test, forward and adjoint for Interp operator for 3d signal
    """
    x = np.ones((par['ny'], par['nx'], par['nt'])) + \
        par['imag'] * np.ones((par['ny'], par['nx'], par['nt']))

    # 1st direction
    Nsub = int(np.round(par['ny'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['ny']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['ny']*par['nx']*par['nt'], iava,
                    dims=(par['ny'], par['nx'], par['nt']), dir=0,
                    kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, Nsub*par['nx']*par['nt'],
                   par['ny']*par['nx']*par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava + 0.3,
                       dims=(par['ny'], par['nx'], par['nt']), dir=0,
                       kind=par['kind'], dtype=par['dtype'])
    assert dottest(Idecop, Nsub * par['nx'] * par['nt'],
                   par['ny'] * par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # repeated indeces
    with pytest.raises(ValueError):
        iava[-2] = 0
        iava[-1] = 0
        _, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava + 0.3,
                      dims=(par['ny'], par['nx'], par['nt']), dir=0,
                      kind=par['kind'], dtype=par['dtype'])

    y = (Iop * x.ravel()).reshape(Nsub, par['nx'], par['nt'])
    ydec = (Idecop * x.ravel()).reshape(Nsub, par['nx'], par['nt'])

    assert_array_almost_equal(y, x[iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[iava])

    # 2nd direction
    Nsub = int(np.round(par['nx'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nx']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava,
                    dims=(par['ny'], par['nx'], par['nt']), dir=1,
                    kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, par['ny'] * Nsub * par['nt'],
                   par['ny'] * par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava + 0.3,
                       dims=(par['ny'], par['nx'], par['nt']), dir=1,
                       kind=par['kind'], dtype=par['dtype'])
    assert dottest(Idecop, par['ny'] * Nsub * par['nt'],
                   par['ny'] * par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    y = (Iop * x.ravel()).reshape(par['ny'], Nsub, par['nt'])
    ydec = (Idecop * x.ravel()).reshape(par['ny'], Nsub, par['nt'])

    assert_array_almost_equal(y, x[:, iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[:, iava])

    # 3rd direction
    Nsub = int(np.round(par['nt'] * perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(par['nt']))[:Nsub])

    # fixed indeces
    Iop, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava,
                    dims=(par['ny'], par['nx'], par['nt']), dir=2,
                    kind=par['kind'], dtype=par['dtype'])
    assert dottest(Iop, par['ny'] * par['nx'] * Nsub,
                   par['ny'] * par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    # decimal indeces
    Idecop, _ = Interp(par['ny'] * par['nx'] * par['nt'], iava + 0.3,
                       dims=(par['ny'], par['nx'], par['nt']), dir=2,
                       kind=par['kind'], dtype=par['dtype'])
    assert dottest(Idecop, par['ny'] * par['nx'] * Nsub,
                   par['ny'] * par['nx'] * par['nt'],
                   complexflag=0 if par['imag'] == 0 else 3)

    y = (Iop * x.ravel()).reshape(par['ny'], par['nx'], Nsub)
    ydec = (Idecop * x.ravel()).reshape(par['ny'], par['nx'], Nsub)

    assert_array_almost_equal(y, x[:, :, iava])
    if par['kind'] == 'nearest':
        assert_array_almost_equal(ydec, x[:, :, iava])
