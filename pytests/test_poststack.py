import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.avo.poststack import PoststackLinearModelling, PoststackInversion

np.random.seed(10)

# params
dt0 = 0.004
ntwav = 41
nsmooth = 50

# 1d model
nt0 = 201
t0 = np.arange(nt0) * dt0
vp = 1200 + np.arange(nt0) + \
     filtfilt(np.ones(5)/5., 1, np.random.normal(0, 80, nt0))
rho = 1000 + vp + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 30, nt0))

m = np.log(vp*rho)
mback = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m)

# 2d model
inputfile = 'testdata/avo/poststack_model.npz'
model = np.load(inputfile)
x, z, m2d = model['x'][::3], model['z'][::3], \
            np.log(model['model'][::3, ::3])
nx, nz = len(x), len(z)

mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m2d, axis=0)
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, mback2d, axis=1)

# wavelet
wav = ricker(t0[:ntwav//2+1], 20)[0]


par1 = {'epsR': None, 'epsI': None,
        'simultaneous': False} # unregularized
par2 = {'epsR': 1e-4, 'epsI': 1e-6,
        'simultaneous': False} # regularized
par3 = {'epsR': None, 'epsI': None,
        'simultaneous': True} # unregularized, simultaneous
par4 = {'epsR': 1e-4, 'epsI': 1e-6,
        'simultaneous': True} # regularized, simultaneous

@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PoststackLinearModelling1d(par):
    """Dot-test, comparison of dense vs lop implementation and
    inversion for PoststackLinearModelling in 1d
    """
    # Dense
    PPop_dense = PoststackLinearModelling(wav, nt0=nt0, explicit=True)
    assert dottest(PPop_dense, nt0, nt0, tol=1e-4)

    # Linear operator
    PPop = PoststackLinearModelling(wav, nt0=nt0, explicit=False)
    assert dottest(PPop, nt0, nt0, tol=1e-4)

    # Compare data
    d = PPop * m.flatten()
    d_dense = PPop_dense * m.T.flatten()
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    for explicit in [True, False]:
        if par['epsR'] is None:
            dict_inv = {}
        else:
            dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'],
                            iter_lim=80)
        minv = PoststackInversion(d, wav, m0=mback, explicit=explicit,
                                  epsR=par['epsR'], epsI=par['epsI'],
                                  simultaneous=par['simultaneous'],
                                  **dict_inv)[0]
        assert np.linalg.norm(m-minv) / np.linalg.norm(minv) < 1e-2


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_PoststackLinearModelling2d(par):
    """Dot-test and inversion for PoststackLinearModelling in 2d
    """

    # Dense
    PPop_dense = PoststackLinearModelling(wav, nt0=nz, spatdims=nx,
                                          explicit=True)
    assert dottest(PPop_dense, nz * nx, nz * nx, tol=1e-4)

    # Linear operator
    PPop = PoststackLinearModelling(wav, nt0=nz, spatdims=nx,
                                    explicit=False)
    assert dottest(PPop, nz * nx, nz * nx, tol=1e-4)

    # Compare data
    d = (PPop * m2d.flatten()).reshape(nz, nx)
    d_dense = (PPop_dense * m2d.flatten()).reshape(nz, nx)
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    for explicit in [True, False]:
        if explicit and not par['simultaneous'] and par['epsR'] is None:
            dict_inv = {}
        elif explicit and not par['simultaneous'] \
                and par['epsR'] is not None:
            dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'],
                            iter_lim=80)
        else:
            dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'],
                            iter_lim=80)

        minv2d = PoststackInversion(d, wav, m0=mback2d, explicit=explicit,
                                    epsR=par['epsR'], epsI=par['epsI'],
                                    simultaneous=par['simultaneous'],
                                    **dict_inv)[0]
        assert np.linalg.norm(m2d - minv2d) / np.linalg.norm(minv2d) < 2e-1
