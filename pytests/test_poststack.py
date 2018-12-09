import pytest

import numpy as np
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.avo.poststack import PoststackLinearModelling, PoststackInversion

np.random.seed(10)

# params
dt0 = 0.004
ntwav = 41
nsmooth = 50

# 1d Model
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
x, z, m2d = model['x'], model['z'], model['model']
nx, nz = len(x), len(z)

nt02d = nz
t02d = np.arange(nt02d) * dt0

# Background model
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m2d, axis=0)
mback2d = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, mback2d, axis=1)

# Wavelet
wav = ricker(t0[:ntwav//2+1], 20)[0]


par1 = {'explicit': True, 'epsR': None,
        'epsI': None, 'simultaneous': False} # explicit operator, unregularized
par2 = {'explicit': True, 'epsR': 1e-4,
        'epsI': 1e-6, 'simultaneous': False} # explicit operator, regularized
par3 = {'explicit': True, 'epsR': None,
        'epsI': None, 'simultaneous': True} # explicit operator, unregularized, simultaneous
par4 = {'explicit': True, 'epsR': 1e-4,
        'epsI': 1e-6, 'simultaneous': True} # explicit operator, regularized, simultaneous
par5 = {'explicit': False, 'epsR': None,
        'epsI': None, 'simultaneous': False} # linear operator, unregularized
par6 = {'explicit': False, 'epsR': 1e-4,
        'epsI': 1e-6, 'simultaneous': False} # linear operator, regularized
par7 = {'explicit': False, 'epsR': None,
        'epsI': None, 'simultaneous': True} # linear operator, unregularized, simultaneous
par8 = {'explicit': False, 'epsR': 1e-4,
        'epsI': 1e-6, 'simultaneous': True} # linear operator, regularized, simultaneous


@pytest.mark.parametrize("par", [(par1), (par2), (par5), (par6)])
def test_PoststackLinearModelling1d(par):
    """Dot-test and inversion for PoststackLinearModelling in 1d
    """
    PPop = PoststackLinearModelling(wav, nt0=nt0,
                                    explicit=par['explicit'])
    print(PPop)
    print(nt0)
    assert dottest(PPop, nt0, nt0, tol=1e-4)

    # Data
    d = PPop * m
    print(d.shape)

    # Inversion
    if par['epsR'] is None:
        dict_inv = {}
    else:
        dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'], iter_lim=80)
    minv = PoststackInversion(d, wav, m0=mback, explicit=par['explicit'],
                              epsR=par['epsR'], epsI=par['epsI'],
                              simultaneous=par['simultaneous'],
                              **dict_inv)[0]
    assert np.linalg.norm(m-minv) / np.linalg.norm(minv) < 1e-2


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7), (par8)])
def test_PoststackLinearModelling2d(par):
    """Dot-test and inversion for PoststackLinearModelling in 2d
    """
    PPop = PoststackLinearModelling(wav, nt0=nz, ndims=nx, explicit=par['explicit'])
    assert dottest(PPop, nz * nx, nz * nx, tol=1e-4)

    # Data
    d = (PPop * m2d.flatten()).reshape(nz, nx)

    # Inversion
    if par['explicit'] and not par['simultaneous'] and par['epsR'] is None:
        dict_inv = {}
    elif par['explicit'] and not par['simultaneous'] and par['epsR'] is not None:
        dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'], iter_lim=80)
    else:
        dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'], iter_lim=80)

    minv2d = PoststackInversion(d, wav, m0=mback2d, explicit=par['explicit'],
                                epsR=par['epsR'], epsI=par['epsI'],
                                simultaneous=par['simultaneous'],
                                **dict_inv)[0]
    assert np.linalg.norm(m2d - minv2d) / np.linalg.norm(minv2d) < 2e-1
