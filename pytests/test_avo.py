import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr

from pylops.utils import dottest
from pylops.avo.avo import zoeppritz_scattering, zoeppritz_pp, zoeppritz_element
from pylops.avo.avo import approx_zoeppritz_pp, akirichards, fatti
from pylops.avo.prestack import AVOLinearModelling


# Create medium parameters for single contrast
vp1, vs1, rho1 = 2200., 1300., 2000  # upper medium
vp0, vs0, rho0 = 2300., 1400., 2100  # lower medium

# Create medium parameters for multiple contrasts
nt0 = 501
dt0 = 0.004
t0 = np.arange(nt0)*dt0
vp = 1200 + np.arange(nt0) + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 80, nt0))
vs = 600 + vp/2 + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 20, nt0))
rho = 1000 + vp + filtfilt(np.ones(5)/5., 1, np.random.normal(0, 30, nt0))
m = (np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)).flatten()

# Angles
ntheta = 21
thetamin, thetamax = 0, 40
theta = np.linspace(thetamin, thetamax, ntheta)

# Parameters
par1 = {'vsvp': 0.5, 'linearization': 'akirich'} # constant vsvp
par2 = {'vsvp': 0.5, 'linearization': 'fatti'}  # constant vsvp
par3 = {'vsvp': vs/vp, 'linearization': 'akirich'} # time-variant vsvp
par4 = {'vsvp': vs/vp, 'linearization': 'fatti'}  # time-variant  vsvp


def test_zoeppritz():
    """Validate zoeppritz using `CREWES Zoeppritz Explorer
    `<https://www.crewes.org/ResearchLinks/ExplorerPrograms/ZE/index.html>`_
    as benchmark
    """
    r_zoep = zoeppritz_scattering(vp1, vs1, rho1, vp0, vs0, rho0, theta[0])
    rpp_zoep = zoeppritz_element(vp1, vs1, rho1, vp0, vs0, rho0, theta[0], element='PdPu')
    rpp_zoep1 = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta[0])

    assert r_zoep.shape == (4, 4, 1)
    assert r_zoep[0, 0] == pytest.approx(0.04658, rel=1e-3)
    assert rpp_zoep == pytest.approx(0.04658, rel=1e-3)
    assert rpp_zoep1 == pytest.approx(0.04658, rel=1e-3)


def test_zoeppritz_and_approx_zeroangle():
    """Validate zoeppritz and approximations at zero incident angle
    """
    #Create composite parameters
    ai0, si0, vpvs0 = vp0 * rho0, vs0 * rho0, vp0 / vs0
    ai1, si1, vpvs1 = vp1 * rho1, vs1 * rho1, vp1 / vs1

    # Zoeppritz
    rpp_zoep = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta[0])
    rpp_zoep_approx = approx_zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta[0])

    # Aki Richards
    rvp = np.log(vp0) - np.log(vp1)
    rvs = np.log(vs0) - np.log(vs1)
    rrho = np.log(rho0) - np.log(rho1)

    G1, G2, G3 = akirichards(theta[0], vs1 / vp1)
    rpp_aki = G1 * rvp + G2 * rvs + G3 * rrho

    # Fatti
    rai = np.log(ai0) - np.log(ai1)
    rsi = np.log(si0) - np.log(si1)

    G1, G2, G3 = fatti(theta[0], vs1 / vp1)
    rpp_fatti = G1 * rai + G2 * rsi + G3 * rrho

    assert_array_almost_equal(rpp_zoep, rpp_zoep_approx, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_aki, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_fatti, decimal=3)


def test_zoeppritz_and_approx_multipleangles():
    """Validate zoeppritz and approximations for set of angles from 0 to 40 degress
    """

    # Create composite parameters
    ai0, si0 = vp0 * rho0, vs0 * rho0
    ai1, si1 = vp1 * rho1, vs1 * rho1

    # Zoeppritz
    rpp_zoep = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta)
    rpp_zoep_approx = approx_zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta)

    # Aki Richards
    rvp = np.log(vp0) - np.log(vp1)
    rvs = np.log(vs0) - np.log(vs1)
    rrho = np.log(rho0) - np.log(rho1)

    G1, G2, G3 = akirichards(theta, vs1 / vp1)
    rpp_aki = G1 * rvp + G2 * rvs + G3 * rrho

    # Fatti
    rai = np.log(ai0) - np.log(ai1)
    rsi = np.log(si0) - np.log(si1)

    G1, G2, G3 = fatti(theta, vs1 / vp1)
    rpp_fatti = G1 * rai + G2 * rsi + G3 * rrho

    assert_array_almost_equal(rpp_zoep, rpp_zoep_approx, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_aki, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_fatti, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_AVOLinearModelling(par):
    """Dot-test and inversion for AVOLinearModelling
    """
    AVOop = AVOLinearModelling(theta, vsvp=par['vsvp'],
                               nt0=nt0, linearization=par['linearization'])
    assert dottest(AVOop, ntheta * nt0, 3 * nt0)

    minv = lsqr(AVOop, AVOop*m, damp=1e-20,
                iter_lim=1000, show=0)[0]
    assert_array_almost_equal(m, minv, decimal=3)
