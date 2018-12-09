"""
Fourier Transform
=================
This example shows how to use the :py:class:`pylops.signalprocessing.FFT` and
:py:class:`pylops.signalprocessing.FFT2D` operators to apply the Fourier Transform
to the model and the inverse Fourier Transform to the data.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by applying the one dimensional FFT to a one dimensional sinusoidal
# signal :math:`d(t)=sin(2 \pi f_0t)` using a time axis of lenght :math:`nt`
# and sampling :math:`dt`
dt = 0.005
nt = 100
t = np.arange(nt)*dt
f0 = 10
nfft = 2**10
d = np.sin(2*np.pi*f0*t)

FFTop = pylops.signalprocessing.FFT(dims=nt, nfft=nfft, sampling=dt)
D = FFTop*d

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t, d, 'k', lw=2, label='True')
axs[0].plot(t, dinv, '--r', lw=2, label='Inverted')
axs[0].legend()
axs[0].set_title('Signal')
axs[1].plot(FFTop.f[:int(FFTop.nfft/2)], np.abs(D[:int(FFTop.nfft/2)]), 'k', lw=2)
axs[1].set_title('Fourier Transform')
axs[1].set_xlim([0, 3*f0])

###############################################################################
# We can also apply the one dimensional FFT to to a two-dimensional
# signal (along one of the first axis)
dt = 0.005
nt, nx = 100, 20
t = np.arange(nt)*dt
f0 = 10
nfft = 2**10
d = np.outer(np.sin(2*np.pi*f0*t), np.arange(nx)+1)

FFTop = pylops.signalprocessing.FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt)
D = FFTop*d.flatten()

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nt, nx)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d, vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(D.reshape(nfft, nx)[:200, :]), cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d-dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

###############################################################################
# Finally we apply the two dimensional FFT to to a two-dimensional signal
dt, dx = 0.005, 5
nt, nx = 100, 201
t = np.arange(nt)*dt
x = np.arange(nx)*dx
f0 = 10
nfft = 2**10
d = np.outer(np.sin(2*np.pi*f0*t), np.arange(nx)+1)

FFTop = pylops.signalprocessing.FFT2D(dims=(nt, nx), nffts=(nfft, nfft), sampling=(dt, dx))
D = FFTop*d.flatten()

dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nt, nx)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d, vmin=-100, vmax=100, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfft), axes=1)[:200, :]), cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv, vmin=-100, vmax=100, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d-dinv, vmin=-100, vmax=100, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()
