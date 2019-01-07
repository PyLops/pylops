import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.avo.prestack import _linearizations, PrestackLinearModelling, \
    PrestackWaveletModelling, PrestackInversion

np.random.seed(10)

# params
dt0 = 0.004
ntwav = 41
ntheta = 7
nsmooth = 50

# angles
thetamin, thetamax = 0, 40
theta = np.linspace(thetamin, thetamax, ntheta)

# 1d model
nt0 = 184
t0 = np.arange(nt0)*dt0
vp = 1200 + np.arange(nt0) + \
     filtfilt(np.ones(5)/5., 1, np.random.normal(0, 40, nt0))
vs = 600 + vp/2 + \
     filtfilt(np.ones(5)/5., 1, np.random.normal(0, 20, nt0))
rho = 1000 + vp + \
      filtfilt(np.ones(5)/5., 1, np.random.normal(0, 30, nt0))

m = np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)
mback = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m, axis=0)


# 2d model
inputfile = 'testdata/avo/poststack_model.npz'
model = np.load(inputfile)
z, x, model = model['z'][::3]/1000., model['x'][::5]/1000., \
              1000*model['model'][::3, ::5]
nx, nz = len(x), len(z)

mvp = model.copy()
mvs = model/2
mrho = model/3+300

m2d = np.log(np.stack((mvp, mvs, mrho), axis=1))
mback2d = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m2d, axis=0)
mback2d = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, mback2d, axis=2)

# wavelet
wav, twav, wavc = ricker(t0[:ntwav // 2 + 1], 20)

# shifted wavelet
wavoff = 10
wav_phase = np.hstack((wav[wavoff:], np.zeros(wavoff)))


# constant vsvp, aki-richards approx, unregularized
par1 = {'vsvp': 0.5, 'linearization': 'akirich',
        'epsR': None, 'epsI': None, 'simultaneous': False}
# constant vsvp, fatti approx, unregularized
par2 = {'vsvp': 0.5, 'linearization': 'fatti',
        'epsR': None, 'epsI': None, 'simultaneous': False}
# time-variant vsvp, aki-richards approx, unregularized
par3 = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'akirich',
        'epsR': None, 'epsI': None, 'simultaneous': False}
# time-variant  vsvp, fatti approx, unregularized
par4 = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'fatti',
        'epsR': None, 'epsI': None, 'simultaneous': False}

# constant vsvp, aki-richards approx, regularized
par1r = {'vsvp': 0.5, 'linearization': 'akirich',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': False}
# constant vsvp, fatti approx, regularized
par2r = {'vsvp': 0.5, 'linearization': 'fatti',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': False}
# time-variant vsvp, aki-richards approx, regularized
par3r = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'akirich',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': False}
# time-variant  vsvp, fatti approx, regularized
par4r = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'fatti',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': False}

# constant vsvp, aki-richards approx, regularized, simultaneous
par1s = {'vsvp': 0.5, 'linearization': 'akirich',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': True}
# constant vsvp, fatti approx, regularized, simultaneous
par2s = {'vsvp': 0.5, 'linearization': 'fatti',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': True}
# time-variant vsvp, aki-richards approx, regularized, simultaneous
par3s = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'akirich',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': True}
# time-variant  vsvp, fatti approx, regularized, simultaneous
par4s = {'vsvp': np.linspace(0.4, 0.6, nt0), 'linearization': 'fatti',
         'epsR': 1e-4, 'epsI': 1e-6, 'simultaneous': True}


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1r), (par2r), (par3r), (par4r),
                                 (par1s), (par2s), (par3s), (par4s)])
def test_PrestackLinearModelling(par):
    """Dot-test, comparison of dense vs lop implementation and
    inversion for PrestackLinearModelling
    """
    #Dense
    PPop_dense = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'], nt0=nt0,
                                         linearization=par['linearization'],
                                         explicit=True)
    assert dottest(PPop_dense, nt0*ntheta,
                   nt0*_linearizations[par['linearization']])

    # Linear operator
    PPop = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'], nt0=nt0,
                                   linearization=par['linearization'],
                                   explicit=False)
    assert dottest(PPop, nt0*ntheta,
                   nt0*_linearizations[par['linearization']])

    # Compare data
    d = PPop * m.ravel()
    d = d.reshape(nt0, ntheta)
    d_dense = PPop_dense * m.T.ravel()
    d_dense = d_dense.reshape(ntheta, nt0).T
    assert_array_almost_equal(d, d_dense, decimal=4)

    # Inversion
    for explicit in [True, False]:
        if par['epsR'] is None:
            dict_inv = {}
        else:
            dict_inv = dict(damp=0 if par['epsI'] is None else par['epsI'],
                            iter_lim=80)
        minv = PrestackInversion(d, theta, wav, m0=mback, explicit=explicit,
                                 epsR=par['epsR'], epsI=par['epsI'],
                                 simultaneous=par['simultaneous'],
                                 **dict_inv)
        assert np.linalg.norm(m - minv) / np.linalg.norm(minv) < 1e-2


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_PrestackWaveletModelling(par):
    """Dot-test and inversion for PrestackWaveletModelling
    """
    # Operators
    Wavestop = \
        PrestackWaveletModelling(m, theta, nwav=ntwav, wavc=wavc,
                                 vsvp=par['vsvp'],
                                 linearization=par['linearization'])
    assert dottest(Wavestop, nt0 * ntheta, ntwav)

    Wavestop_phase = \
        PrestackWaveletModelling(m, theta, nwav=ntwav, wavc=wavc,
                                 vsvp=par['vsvp'],
                                 linearization=par['linearization'])
    assert dottest(Wavestop_phase, nt0 * ntheta, ntwav)

    # Create data
    d = (Wavestop * wav).reshape(ntheta, nt0).T
    d_phase = (Wavestop_phase * wav_phase).reshape(ntheta, nt0).T

    # Estimate wavelet
    wav_est = Wavestop / d.T.ravel()
    wav_phase_est = Wavestop_phase / d_phase.T.ravel()

    assert_array_almost_equal(wav, wav_est, decimal=3)
    assert_array_almost_equal(wav_phase, wav_phase_est, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1r), (par2r), (par3r), (par4r),
                                 (par1s), (par2s), (par3s), (par4s)])
def test_PrestackLinearModelling2d(par):
    """Dot-test and inversion for PoststackLinearModelling in 2d
    """
    nm = _linearizations[par['linearization']]
    # Dense
    PPop_dense = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'],
                                         nt0=nz, spatdims=(nx, ),
                                         linearization=par['linearization'],
                                         explicit=True)
    assert dottest(PPop_dense, nz * ntheta * nx, nz * nm * nx)

    # Linear operator
    PPop = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'],
                                   nt0=nz, spatdims=(nx,),
                                   linearization=par['linearization'],
                                   explicit=False)
    assert dottest(PPop_dense, nz * ntheta * nx, nz * nm * nx)

    # Compare data
    d = (PPop * m2d.ravel()).reshape(nz, ntheta, nx)
    d_dense = (PPop_dense * m2d.swapaxes(0, 1).ravel()).\
        reshape(ntheta, nz, nx).swapaxes(0, 1)
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

        minv2d = PrestackInversion(d, theta, wav, m0=mback2d,
                                   explicit=explicit,
                                   epsR=par['epsR'], epsI=par['epsI'],
                                   simultaneous=par['simultaneous'],
                                   **dict_inv)
        assert np.linalg.norm(m2d - minv2d) / np.linalg.norm(minv2d) < 2e-1
