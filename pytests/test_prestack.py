import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.avo.prestack import PrestackLinearModelling, PrestackWaveletModelling


# Create medium parameters for multiple contrasts
nt0 = 201
dt0 = 0.004
t0 = np.arange(nt0)*dt0
vp = 1200 + np.arange(nt0) + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 80, nt0))
vs = 600 + vp/2 + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 20, nt0))
rho = 1000 + vp + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 30, nt0))
m = np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)

# Angles
ntheta = 7
thetamin, thetamax = 0, 40
theta = np.linspace(thetamin, thetamax, ntheta)

# Wavelet
ntwav = 41
wav, twav, wavc = ricker(t0[:ntwav // 2 + 1], 20)

# Shifted wavelet
wavoff = 10
wav_phase = np.hstack((wav[wavoff:], np.zeros(wavoff)))


# Parameters
par1 = {'vsvp': 0.5, 'linearization': 'akirich'} # constant vsvp, aki-richards approx
par2 = {'vsvp': 0.5, 'linearization': 'fatti'}  # constant vsvp, fatti approx
par3 = {'vsvp': vs/vp, 'linearization': 'akirich'} # time-variant vsvp, aki-richards approx
par4 = {'vsvp': vs/vp, 'linearization': 'fatti'}  # time-variant  vsvp, fatti approx


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_PrestackLinearModelling(par):
    """Dot-test and comparison of dense vs lop implementation for PrestackLinearModelling
    """
    # Dense operator
    PPop_dense = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'],
                                         nt0=nt0, linearization=par['linearization'],
                                         explicit=True)
    assert dottest(PPop_dense, nt0 * ntheta, nt0 * 3)

    # Linear operator
    PPop = PrestackLinearModelling(wav, theta, vsvp=par['vsvp'],
                                   nt0=nt0, linearization=par['linearization'])
    assert dottest(PPop, nt0 * ntheta, nt0 * 3)

    # Compare data
    d = PPop * m.flatten()
    d = d.reshape(nt0, ntheta)
    d_dense = PPop_dense * m.T.flatten()
    d_dense = d_dense.reshape(ntheta, nt0).T
    assert_array_almost_equal(d, d_dense, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_PrestackWaveletModelling(par):
    """Dot-test and inversion for PrestackWaveletModelling
    """
    # Operators
    Wavestop = PrestackWaveletModelling(m, theta, nwav=ntwav, wavc=wavc,
                                        vsvp=par['vsvp'], linearization=par['linearization'])
    assert dottest(Wavestop, nt0 * ntheta, ntwav)

    Wavestop_phase = PrestackWaveletModelling(m, theta, nwav=ntwav, wavc=wavc,
                                              vsvp=par['vsvp'], linearization=par['linearization'])
    assert dottest(Wavestop_phase, nt0 * ntheta, ntwav)

    # Create data
    d = (Wavestop * wav).reshape(ntheta, nt0).T
    d_phase = (Wavestop_phase * wav_phase).reshape(ntheta, nt0).T

    # Estimate wavelet
    wav_est = Wavestop / d.T.flatten()
    wav_phase_est = Wavestop_phase / d_phase.T.flatten()

    assert_array_almost_equal(wav, wav_est, decimal=3)
    assert_array_almost_equal(wav_phase, wav_phase_est, decimal=3)
