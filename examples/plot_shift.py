r"""
Shift
=====

This example shows how to use the :py:class:`pylops.signalprocessing.Shift`
operator to apply fractional delay to an input signal. Whilst this operator
acts on 1D signals it can also be applied on any multi-dimensional signal on
a specific direction of it.
"""

import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

###############################################################################
# Let's start with a 1D example. Define the input parameters: number of samples
# of input signal (``nt``), sampling step (``dt``) as well as the input
# signal which will be equal to a ricker wavelet:
nt = 127
dt = 0.004
t = np.arange(nt) * dt
ntwav = 41

wav = pylops.utils.wavelets.ricker(t[:ntwav], f0=20)[0]
wav = np.pad(wav, [0, nt - len(wav)])
WAV = np.fft.rfft(wav, n=nt)

###############################################################################
# We can shift this wavelet by :math:`5.5*dt`:
shift = 5.5 * dt
Op = pylops.signalprocessing.Shift(nt, shift, sampling=dt, real=True, dtype=np.float64)
wavshift = Op * wav
wavshiftback = Op.H * wavshift

plt.figure(figsize=(10, 3))
plt.plot(t, wav, "k", lw=2, label="Original")
plt.plot(t, wavshift, "r", lw=2, label="Shifted")
plt.plot(t, wavshiftback, "--b", lw=2, label="Adjoint")
plt.axvline(t[ntwav - 1], color="k")
plt.axvline(t[ntwav - 1] + shift, color="r")
plt.xlim(0, 0.3)
plt.legend()
plt.title("1D Shift")
plt.tight_layout()

###############################################################################
# We can repeat the same exercise for a 2D signal and perform the shift
# along the first and second dimensions.

shift = 10.5 * dt

# 1st dir
wav2d = np.outer(wav, np.ones(10))
Op = pylops.signalprocessing.Shift(
    (nt, 10), shift, axis=0, sampling=dt, real=True, dtype=np.float64
)
wav2dshift = (Op * wav2d.ravel()).reshape(nt, 10)
wav2dshiftback = (Op.H * wav2dshift.ravel()).reshape(nt, 10)

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(wav2d, cmap="gray")
axs[0].axis("tight")
axs[0].set_title("Original")
axs[1].imshow(wav2dshift, cmap="gray")
axs[1].set_title("Shifted")
axs[1].axis("tight")
axs[2].imshow(wav2dshiftback, cmap="gray")
axs[2].set_title("Adjoint")
axs[2].axis("tight")
fig.tight_layout()

# 2nd dir
wav2d = np.outer(wav, np.ones(10)).T
Op = pylops.signalprocessing.Shift(
    (10, nt), shift, axis=1, sampling=dt, real=True, dtype=np.float64
)
wav2dshift = (Op * wav2d.ravel()).reshape(10, nt)
wav2dshiftback = (Op.H * wav2dshift.ravel()).reshape(10, nt)

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(wav2d, cmap="gray")
axs[0].axis("tight")
axs[0].set_title("Original")
axs[1].imshow(wav2dshift, cmap="gray")
axs[1].set_title("Shifted")
axs[1].axis("tight")
axs[2].imshow(wav2dshiftback, cmap="gray")
axs[2].set_title("Adjoint")
axs[2].axis("tight")
fig.tight_layout()
