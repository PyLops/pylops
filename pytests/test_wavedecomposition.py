import pytest
import numpy as np

from pylops.utils.wavelets import ricker
from pylops.utils.seismicevents import makeaxis, hyperbolic2d, hyperbolic3d
from pylops.waveeqprocessing.wavedecomposition import UpDownComposition2D, \
    UpDownComposition3D, WavefieldDecomposition, PressureToVelocity

# params
PAR = {'ox': -100, 'dx': 10, 'nx': 21,
       'oy': -50, 'dy': 10, 'ny': 11,
       'ot': 0, 'dt': 0.004, 'nt': 50,
       'f0': 40}

par1 = PAR.copy() # analytical
par1['kind'] = 'analytical'
par2 = PAR.copy() # inverse
par2['kind'] = 'inverse'

# separation params
vel_sep = 1000.0 # velocity at separation level
rho_sep = 1000.0 # density at separation level
critical = 0.9
ntaper = 41
nfftf = 2**8
nfftk = 2**7

# axes and wavelet
t, t2, x, y = makeaxis(PAR)
wav = ricker(t[:41], f0=PAR['f0'])[0]


@pytest.fixture(scope="module")
def create_data2D():
    """Create 2d dataset
    """
    t0_plus = np.array([0.05, 0.12])
    t0_minus = t0_plus + 0.04
    vrms = np.array([1400., 1800.])
    amp = np.array([1., -0.6])

    _, p2d_minus = hyperbolic2d(x, t, t0_minus, vrms, amp, wav)
    _, p2d_plus = hyperbolic2d(x, t, t0_plus, vrms, amp, wav)

    UPop = \
        UpDownComposition2D(PAR['nt'], PAR['nx'],
                            PAR['dt'], PAR['dx'],
                            rho_sep, vel_sep,
                            nffts=(nfftk, nfftf),
                            critical=critical * 100.,
                            ntaper=ntaper,
                            dtype='complex128')

    d2d = UPop * np.concatenate((p2d_plus.ravel(),
                                 p2d_minus.ravel())).ravel()
    d2d = np.real(d2d.reshape(2 * PAR['nx'], PAR['nt']))
    p2d, vz2d = d2d[:PAR['nx']], d2d[PAR['nx']:]
    return p2d, vz2d, p2d_minus, p2d_plus


@pytest.fixture(scope="module")
def create_data3D():
    """Create 3d dataset
    """
    t0_plus = np.array([0.05, 0.12])
    t0_minus = t0_plus + 0.04
    vrms = np.array([1400., 1800.])
    amp = np.array([1., -0.6])

    _, p3d_minus = hyperbolic3d(x, y, t, t0_minus, vrms, vrms, amp, wav)
    _, p3d_plus = hyperbolic3d(x, y, t, t0_plus, vrms, vrms, amp, wav)

    UPop = \
        UpDownComposition3D(PAR['nt'], (PAR['ny'], PAR['nx']),
                            PAR['dt'], (PAR['dy'], PAR['dx']),
                            rho_sep, vel_sep,
                            nffts=(nfftk, nfftk, nfftf),
                            critical=critical * 100.,
                            ntaper=ntaper,
                            dtype='complex128')

    d3d = UPop * np.concatenate((p3d_plus.ravel(),
                                 p3d_minus.ravel())).ravel()
    d3d = np.real(d3d.reshape(2 * PAR['ny'], PAR['nx'], PAR['nt']))
    p3d, vz3d = d3d[:PAR['ny']], d3d[PAR['ny']:]
    return p3d, vz3d, p3d_minus, p3d_plus


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_WavefieldDecomposition2D(par, create_data2D):
    """WavefieldDecomposition and PressureToVelocity reconstruction of 2d data
    """
    p2d, vz2d, p2d_minus, p2d_plus = create_data2D

    # decomposition
    p2d_minus_est, p2d_plus_est = \
        WavefieldDecomposition(p2d, vz2d, par['nt'],
                               par['nx'],
                               par['dt'], par['dx'],
                               rho_sep, vel_sep,
                               nffts=(nfftk, nfftf),
                               kind=par['kind'],
                               critical=critical * 100,
                               ntaper=ntaper,
                               dottest=True,
                               dtype='complex128',
                               **dict(damp=1e-10,
                                      iter_lim=10))
    assert np.linalg.norm(p2d_minus_est - p2d_minus) / \
           np.linalg.norm(p2d_minus) < 2e-1
    assert np.linalg.norm(p2d_plus_est - p2d_plus) / \
           np.linalg.norm(p2d_plus) < 2e-1

    # reconstruction
    PtoVop = PressureToVelocity(par['nt'], par['nx'], par['dt'], par['dx'],
                                rho_sep, vel_sep, nffts=(nfftk, nfftf),
                                critical=critical * 100., ntaper=ntaper,
                                topressure=False)
    vz2d_plus_est = \
        (PtoVop * p2d_plus_est.ravel()).reshape(par['nx'], par['nt'])
    vz2d_minus_est = \
        (PtoVop * p2d_minus_est.ravel()).reshape(par['nx'], par['nt'])
    vz2d_est = np.real(vz2d_plus_est - vz2d_minus_est)

    assert np.linalg.norm(vz2d_est - vz2d) / np.linalg.norm(vz2d) < 2e-1


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_WavefieldDecomposition3D(par, create_data3D):
    """WavefieldDecomposition and PressureToVelocity reconstruction of 3d data
    """
    p3d, vz3d, p3d_minus, p3d_plus = create_data3D

    # decomposition
    p3d_minus_est, p3d_plus_est = \
        WavefieldDecomposition(p3d, vz3d, par['nt'],
                               (par['ny'], par['nx']),
                               par['dt'], (par['dy'], par['dx']),
                               rho_sep, vel_sep,
                               nffts=(nfftk, nfftk, nfftf),
                               kind=par['kind'],
                               critical=critical * 100,
                               ntaper=ntaper,
                               dottest=True,
                               dtype='complex128',
                               **dict(damp=1e-10,
                                      iter_lim=10, show=2))
    assert np.linalg.norm(p3d_minus_est - p3d_minus) / \
           np.linalg.norm(p3d_minus) < 3e-1
    assert np.linalg.norm(p3d_plus_est - p3d_plus) / \
           np.linalg.norm(p3d_plus) < 3e-1

    # reconstruction
    PtoVop = PressureToVelocity(par['nt'], (par['ny'], par['nx']),
                                par['dt'], (par['dy'], par['dx']),
                                rho_sep, vel_sep, nffts=(nfftk, nfftk, nfftf),
                                critical=critical * 100., ntaper=ntaper,
                                topressure=False)
    vz3d_plus_est = \
        (PtoVop * p3d_plus_est.ravel()).reshape(par['ny'], par['nx'],
                                                par['nt'])
    vz3d_minus_est = \
        (PtoVop * p3d_minus_est.ravel()).reshape(par['ny'], par['nx'],
                                                 par['nt'])
    vz3d_est = np.real(vz3d_plus_est - vz3d_minus_est)

    assert np.linalg.norm(vz3d_est - vz3d) / np.linalg.norm(vz3d) < 3e-1
