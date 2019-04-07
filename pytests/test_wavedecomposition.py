import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.signal import filtfilt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.utils.seismicevents import makeaxis, hyperbolic2d
from pylops.signalprocessing import FFT2D
from pylops.waveeqprocessing.wavedecomposition import UpDownComposition2D, \
    WavefieldDecomposition

# params
PAR = {'ox': -100, 'dx': 5, 'nx': 41,
       'ox': -140, 'dx': 5, 'nx': 51,
       'ot': 0, 'dt': 0.004, 'nt': 100,
       'f0': 40}

# analytical
par1 = PAR.copy()
par1['kind'] = 'analytical'
# inverse
par2 = PAR.copy()
par2['kind'] = 'inverse'

# separation params
critical = 1.1
ntaper = 51
nfft=2**10

# axes and wavelet
t, t2, x, y = makeaxis(PAR)
wav = ricker(t[:41], f0=PAR['f0'])[0]

# 2d data
t0_plus = np.array([0.2, 0.5, 0.7])
t0_minus = t0_plus + 0.04
vrms = np.array([1400., 1500., 2000.])
amp = np.array([1., -0.6, 0.5])
vel_sep = 1000.0 # velocity at separation level
rho_sep = 1000.0 # density at separation level

_, p2d_minus = hyperbolic2d(x, t, t0_minus, vrms, amp, wav)
_, p2d_plus = hyperbolic2d(x, t, t0_plus, vrms, amp, wav)

FFTop = FFT2D(dims=[PAR['nx'], PAR['nt']],
              nffts=[nfft, nfft],
              sampling=[PAR['dx'], PAR['dt']])

[Kx, F] = np.meshgrid(FFTop.f1, FFTop.f2, indexing='ij')
k = F/vel_sep
Kz = np.sqrt((k**2-Kx**2).astype(np.complex))
Kz[np.isnan(Kz)] = 0
OBL=rho_sep*(np.abs(F)/Kz)
OBL[Kz==0]=0

mask = np.abs(Kx)<critical*np.abs(F)/vel_sep
OBL *= mask
OBL = filtfilt(np.ones(ntaper)/float(ntaper), 1, OBL, axis=0)
OBL = filtfilt(np.ones(ntaper)/float(ntaper), 1, OBL, axis=1)

UPop = \
    UpDownComposition2D(PAR['nt'], PAR['nx'],
                        PAR['dt'], PAR['dx'],
                        rho_sep, vel_sep,
                        nffts=(nfft, nfft),
                        critical=critical*100.,
                        ntaper=ntaper,
                        dtype='complex128')

d2d = UPop * np.concatenate((p2d_plus.flatten(),
                             p2d_minus.flatten())).flatten()
d2d = np.real(d2d.reshape(2*PAR['nx'], PAR['nt']))
p2d, vz2d = d2d[:PAR['nx']], d2d[PAR['nx']:]


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_WavefieldDecompositionD(par):
    """WavefieldDecomposition operator of 2d data
    """
    p2d_minus_est, p2d_plus_est= \
        WavefieldDecomposition(p2d, vz2d, par['nt'],
                               par['nx'],
                               par['dt'], par['dx'],
                               rho_sep, vel_sep,
                               nffts=(nfft, nfft),
                               kind=par['kind'],
                               critical=critical * 100,
                               ntaper=ntaper,
                               dottest=True,
                               dtype='complex128',
                               **dict(damp=1e-10,
                                      iter_lim=20))

    assert np.linalg.norm(p2d_minus_est - p2d_minus) / \
           np.linalg.norm(p2d_minus) < 2e-1
    assert np.linalg.norm(p2d_plus_est - p2d_plus) / \
           np.linalg.norm(p2d_plus) < 2e-1
