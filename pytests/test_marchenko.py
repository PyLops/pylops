import pytest

import numpy as np
from scipy.signal import convolve
from pylops.waveeqprocessing.marchenko import Marchenko

# Test data
inputfile = 'testdata/marchenko/input.npz'

# Parameters
vel = 2400.0  # velocity
toff = 0.045  # direct arrival time shift
nsmooth = 10  # time window smoothing
nfmax = 1000  # max frequency for MDC (#samples)

# Input data
inputdata = np.load(inputfile)

# Receivers
r = inputdata['r']
nr = r.shape[1]
dr = r[0, 1] - r[0, 0]

# Sources
s = inputdata['s']
ns = s.shape[1]

# Virtual points
vs = inputdata['vs']

# Multiple virtual points
vs_multi = [np.arange(-1, 2)*100 + vs[0],
            np.ones(3)*vs[1]]

# Density model
rho = inputdata['rho']
z, x = inputdata['z'], inputdata['x']

# Reflection data and subsurface fields
R = inputdata['R']
R = np.swapaxes(R, 0, 1)

gsub = inputdata['Gsub']
g0sub = inputdata['G0sub']
wav = inputdata['wav']
wav_c = np.argmax(wav)

t = inputdata['t']
ot, dt, nt = t[0], t[1] - t[0], len(t)

gsub = np.apply_along_axis(convolve, 0, gsub, wav, mode='full')
gsub = gsub[wav_c:][:nt]
g0sub = np.apply_along_axis(convolve, 0, g0sub, wav, mode='full')
g0sub = g0sub[wav_c:][:nt]

# Direct arrival window
trav = np.sqrt((vs[0] - r[0]) ** 2 + (vs[1] - r[1]) ** 2) / vel
trav_multi = np.sqrt((vs_multi[0]-r[0][:, np.newaxis])**2+(vs_multi[1]-r[1][:, np.newaxis])**2)/vel


# Create Rs in frequency domain
Rtwosided = np.concatenate((np.zeros((nr, ns, nt-1)), R), axis=-1)
R1twosided = np.concatenate((np.flip(R, axis=-1),
                             np.zeros((nr, ns, nt - 1))), axis=-1)

Rtwosided_fft = np.fft.rfft(Rtwosided, 2*nt-1, axis=-1) / np.sqrt(2*nt-1)
Rtwosided_fft = Rtwosided_fft[..., :nfmax]
R1twosided_fft = np.fft.rfft(R1twosided, 2*nt-1, axis=-1) / np.sqrt(2*nt-1)
R1twosided_fft = R1twosided_fft[..., :nfmax]

par1 = {'niter': 10}


@pytest.mark.parametrize("par", [(par1)])
def test_Marchenko_freq(par):
    """Solve marchenko equations using input Rs in frequency domain
    """
    MarchenkoWM = Marchenko(Rtwosided_fft, R1=R1twosided_fft, nt=nt, dt=dt,
                            dr=dr, nfmax=nfmax, wav=wav,
                            toff=toff, nsmooth=nsmooth)

    _, _, _, g_inv_minus, g_inv_plus = \
        MarchenkoWM.apply_onepoint(trav, G0=g0sub.T, rtm=True, greens=True,
                                   dottest=True, **dict(iter_lim=par['niter'], show=True))
    ginvsub = (g_inv_minus + g_inv_plus)[:, nt-1:].T
    ginvsub_norm = ginvsub / ginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm-ginvsub_norm) / np.linalg.norm(gsub_norm) < 1e-1


@pytest.mark.parametrize("par", [(par1)])
def test_Marchenko_time(par):
    """Solve marchenko equations using input Rs in time domain
    """
    MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                            toff=toff, nsmooth=nsmooth)

    _, _, _, g_inv_minus, g_inv_plus = \
        MarchenkoWM.apply_onepoint(trav, G0=g0sub.T, rtm=True, greens=True,
                                   dottest=True, **dict(iter_lim=par['niter'], show=True))
    ginvsub = (g_inv_minus + g_inv_plus)[:, nt - 1:].T
    ginvsub_norm = ginvsub / ginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm - ginvsub_norm) / np.linalg.norm(gsub_norm) < 1e-1


@pytest.mark.parametrize("par", [(par1)])
def test_Marchenko_time_ana(par):
    """Solve marchenko equations using input Rs in time domain and analytical direct wave
    """
    MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                            toff=toff, nsmooth=nsmooth)

    _, _, g_inv_minus, g_inv_plus = \
        MarchenkoWM.apply_onepoint(trav, nfft=2**11, rtm=False, greens=True,
                                   dottest=True, **dict(iter_lim=par['niter'], show=True))
    ginvsub = (g_inv_minus + g_inv_plus)[:, nt-1:].T
    ginvsub_norm = ginvsub / ginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm-ginvsub_norm) / np.linalg.norm(gsub_norm) < 1e-1


@pytest.mark.parametrize("par", [(par1)])
def test_Marchenko_timemulti_ana(par):
    """Solve marchenko equations using input Rs in time domain with multiple points
    """
    MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                            toff=toff, nsmooth=nsmooth)

    _, _, g_inv_minus, g_inv_plus = \
        MarchenkoWM.apply_multiplepoints(trav_multi, nfft=2**11, rtm=False, greens=True,
                                         dottest=True, **dict(iter_lim=par['niter'], show=True))
    ginvsub = (g_inv_minus + g_inv_plus)[:, 1, nt - 1:].T
    ginvsub_norm = ginvsub / ginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm - ginvsub_norm) / np.linalg.norm(gsub_norm) < 1e-1
