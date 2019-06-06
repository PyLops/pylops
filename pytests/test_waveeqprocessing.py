import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.utils.seismicevents import makeaxis, linear2d, linear3d
from pylops.waveeqprocessing.mdd import MDC, MDD

PAR = {'ox': 0, 'dx': 2, 'nx': 10,
       'oy': 0, 'dy': 2, 'ny': 20,
       'ot': 0, 'dt': 0.004, 'nt': 401,
       'f0': 20}

# nt odd, single-sided, full fft
par1 = PAR.copy()
par1['twosided'] = False
par1['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt odd, double-sided, full fft
par2 = PAR.copy()
par2['twosided'] = True
par2['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt odd, single-sided, truncated fft
par3 = PAR.copy()
par3['twosided'] = False
par3['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt odd, double-sided, truncated fft
par4 = PAR.copy()
par4['twosided'] = True
par4['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt even, single-sided, full fft
par4 = PAR.copy()
par4['nt'] -= 1
par4['twosided'] = False
par4['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt even, double-sided, full fft
par5 = PAR.copy()
par5['nt'] -= 1
par5['twosided'] = True
par5['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt even, single-sided, truncated fft
par6 = PAR.copy()
par6['nt'] -= 1
par6['twosided'] = False
par6['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt even, double-sided, truncated fft
par7 = PAR.copy()
par7['nt'] -= 1
par7['twosided'] = True
par7['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7)])
def test_MDC_1virtualsource(par):
    """Dot-test and inversion for MDC operator of 1 virtual source
    """
    if par['twosided']:
        par['nt2'] = 2*par['nt'] - 1
    else:
        par['nt2'] = par['nt']
    v = 1500
    t0_m = 0.2
    theta_m = 0
    amp_m = 1.

    t0_G = (0.1, 0.2, 0.3)
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1., 0.6, 2.)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par['f0'])[0]

    # Generate model
    _, mwav = linear2d(x, t, v, t0_m, theta_m, amp_m, wav)
    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par['twosided']:
        mwav = np.concatenate((np.zeros((par['nx'], par['nt'] - 1)), mwav), axis=-1)
        Gwav = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt'] - 1)), Gwav), axis=-1)

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par['nt2'], axis=-1)
    Gwav_fft = Gwav_fft[..., :par['nfmax']]

    MDCop = MDC(Gwav_fft, nt=par['nt2'], nv=1,
                dt=par['dt'], dr=par['dx'],
                twosided=par['twosided'], dtype='float32')
    dottest(MDCop, par['nt2']*par['ny'], par['nt2']*par['nx'])

    # Create data
    d = MDCop * mwav.flatten()
    d = d.reshape(par['ny'], par['nt2'])

    # Apply mdd function
    minv = MDD(Gwav[:, :, par['nt']-1:] if par['twosided'] else Gwav,
               d[:, par['nt']-1:] if par['twosided'] else d,
               dt=par['dt'], dr=par['dx'], nfmax=par['nfmax'],
               twosided=par['twosided'], adjoint=False, psf=False, dtype='complex64',
               dottest=False,
               **dict(damp=1e-10, iter_lim=50, show=1))
    assert_array_almost_equal(mwav, minv, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7)])
def test_MDC_Nvirtualsources(par):
    """Dot-test and inversion for MDC operator of N virtual source
    """
    if par['twosided']:
        par['nt2'] = 2*par['nt'] - 1
    else:
        par['nt2'] = par['nt']
    v = 1500
    t0_m = 0.2
    theta_m = 0
    phi_m = 0
    amp_m = 1.

    t0_G = (0.1, 0.2, 0.3)
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1., 0.6, 2.)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par['f0'])[0]

    # Generate model
    _, mwav = linear3d(x, x, t, v, t0_m, theta_m, phi_m, amp_m, wav)

    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par['twosided']:
        mwav = np.concatenate((np.zeros((par['nx'], par['nx'], par['nt'] - 1)), mwav), axis=-1)
        Gwav = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt'] - 1)), Gwav), axis=-1)

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par['nt2'], axis=-1)
    Gwav_fft = Gwav_fft[..., :par['nfmax']]

    MDCop = MDC(Gwav_fft, nt=par['nt2'], nv=par['nx'],
                dt=par['dt'], dr=par['dx'], twosided=par['twosided'], dtype='float32')
    dottest(MDCop, par['nt2']*par['ny']*par['nx'], par['nt2']*par['nx']*par['nx'])

    # Create data
    d = MDCop * mwav.flatten()
    d = d.reshape(par['ny'], par['nx'], par['nt2'])

    # Apply mdd function
    minv = MDD(Gwav[:, :, par['nt']-1:] if par['twosided'] else Gwav,
               d[:, :, par['nt']-1:] if par['twosided'] else d,
               dt=par['dt'], dr=par['dx'], nfmax=par['nfmax'], twosided=par['twosided'],
               adjoint=False, psf=False, dtype='complex64', dottest=False,
               **dict(damp=1e-10, iter_lim=50, show=1))
    assert_array_almost_equal(mwav, minv, decimal=2)
