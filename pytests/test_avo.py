import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest
from scipy.signal import filtfilt

from pylops.avo.avo import (
    akirichards,
    approx_zoeppritz_pp,
    fatti,
    zoeppritz_element,
    zoeppritz_pp,
    zoeppritz_scattering,
)
from pylops.avo.prestack import AVOLinearModelling
from pylops.optimization.basic import lsqr
from pylops.utils import dottest
from pylops.utils.backend import to_numpy

np.random.seed(0)

# Create medium parameters for single contrast
vp1, vs1, rho1 = 2200.0, 1300.0, 2000  # upper medium
vp0, vs0, rho0 = 2300.0, 1400.0, 2100  # lower medium

# Create medium parameters for multiple contrasts
nt0 = 201
dt0 = 0.004
t0 = npp.arange(nt0) * dt0
vp = (
    1200
    + npp.arange(nt0)
    + filtfilt(npp.ones(5) / 5.0, 1, npp.random.normal(0, 80, nt0))
)
vs = 600 + vp / 2 + filtfilt(npp.ones(5) / 5.0, 1, npp.random.normal(0, 20, nt0))
rho = 1000 + vp + filtfilt(npp.ones(5) / 5.0, 1, npp.random.normal(0, 30, nt0))
m = npp.stack((npp.log(vp), npp.log(vs), npp.log(rho)), axis=1).ravel()

# Angles
ntheta = 21
thetamin, thetamax = 0, 40
theta = npp.linspace(thetamin, thetamax, ntheta)

# Parameters
par1 = {"vsvp": 0.5, "linearization": "akirich"}  # constant vsvp
par2 = {"vsvp": 0.5, "linearization": "fatti"}  # constant vsvp
par3 = {"vsvp": vs / vp, "linearization": "akirich"}  # time-variant vsvp
par4 = {"vsvp": vs / vp, "linearization": "fatti"}  # time-variant  vsvp


def test_zoeppritz():
    """Validate zoeppritz using `CREWES Zoeppritz Explorer
    `<https://www.crewes.org/ResearchLinks/ExplorerPrograms/ZE/index.html>`_
    as benchmark
    """
    r_zoep = zoeppritz_scattering(vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)[:1])
    rpp_zoep = zoeppritz_element(
        vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)[:1], element="PdPu"
    )
    rpp_zoep1 = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)[:1])

    assert r_zoep.shape == (4, 4, 1)
    assert to_numpy(r_zoep[0, 0, 0]) == pytest.approx(0.04658, rel=1e-3)
    assert to_numpy(rpp_zoep) == pytest.approx(0.04658, rel=1e-3)
    assert to_numpy(rpp_zoep1) == pytest.approx(0.04658, rel=1e-3)


def test_zoeppritz_and_approx_zeroangle():
    """Validate zoeppritz and approximations at zero incident angle"""
    # Create composite parameters
    ai0, si0, _ = vp0 * rho0, vs0 * rho0, vp0 / vs0
    ai1, si1, _ = vp1 * rho1, vs1 * rho1, vp1 / vs1

    # Zoeppritz
    rpp_zoep = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)[:1])
    rpp_zoep_approx = approx_zoeppritz_pp(
        vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)[:1]
    )

    # Aki Richards
    rvp = np.asarray(np.log(vp0) - np.log(vp1))
    rvs = np.asarray(np.log(vs0) - np.log(vs1))
    rrho = np.asarray(np.log(rho0) - np.log(rho1))

    G1, G2, G3 = akirichards(np.asarray(theta)[:1], vs1 / vp1)
    rpp_aki = G1 * rvp + G2 * rvs + G3 * rrho

    # Fatti
    rai = np.asarray(np.log(ai0) - np.log(ai1))
    rsi = np.asarray(np.log(si0) - np.log(si1))

    G1, G2, G3 = fatti(np.asarray(theta)[:1], vs1 / vp1)
    rpp_fatti = G1 * rai + G2 * rsi + G3 * rrho

    assert_array_almost_equal(rpp_zoep, rpp_zoep_approx, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_aki, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_fatti, decimal=3)


def test_zoeppritz_and_approx_multipleangles():
    """Validate zoeppritz and approximations for set of angles from 0 to 40 degress"""

    # Create composite parameters
    ai0, si0 = vp0 * rho0, vs0 * rho0
    ai1, si1 = vp1 * rho1, vs1 * rho1

    # Zoeppritz
    rpp_zoep = zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta))
    rpp_zoep_approx = approx_zoeppritz_pp(
        vp1, vs1, rho1, vp0, vs0, rho0, np.asarray(theta)
    )

    # Aki Richards
    rvp = np.asarray(np.log(vp0) - np.log(vp1))
    rvs = np.asarray(np.log(vs0) - np.log(vs1))
    rrho = np.asarray(np.log(rho0) - np.log(rho1))

    G1, G2, G3 = akirichards(np.asarray(theta), vs1 / vp1)
    rpp_aki = G1 * rvp + G2 * rvs + G3 * rrho

    # Fatti
    rai = np.asarray(np.log(ai0) - np.log(ai1))
    rsi = np.asarray(np.log(si0) - np.log(si1))

    G1, G2, G3 = fatti(np.asarray(theta), vs1 / vp1)
    rpp_fatti = G1 * rai + G2 * rsi + G3 * rrho

    assert_array_almost_equal(rpp_zoep, rpp_zoep_approx, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_aki, decimal=3)
    assert_array_almost_equal(rpp_zoep, rpp_fatti, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_AVOLinearModelling(par):
    """Dot-test and inversion for AVOLinearModelling"""
    AVOop = AVOLinearModelling(
        np.asarray(theta),
        vsvp=par["vsvp"] if isinstance(par["vsvp"], float) else np.asarray(par["vsvp"]),
        nt0=nt0,
        linearization=par["linearization"],
    )
    assert dottest(AVOop, ntheta * nt0, 3 * nt0, backend=backend)

    minv = lsqr(
        AVOop,
        AVOop * np.asarray(m),
        x0=np.zeros_like(m),
        damp=1e-20,
        niter=1000,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(m, minv, decimal=3)
