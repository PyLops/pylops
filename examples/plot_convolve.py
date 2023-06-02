"""
Convolution
===========
This example shows how to use the :py:class:`pylops.signalprocessing.Convolve1D`,
:py:class:`pylops.signalprocessing.Convolve2D` and
:py:class:`pylops.signalprocessing.ConvolveND` operators to perform convolution
between two signals.

Such operators can be used in the forward model of several common application
in signal processing that require filtering of an input signal for the
instrument response. Similarly, removing the effect of the instrument
response from signal is equivalent to solving linear system of equations
based on Convolve1D, Convolve2D or ConvolveND operators.
This problem is generally referred to as *Deconvolution*.

A very practical example of deconvolution can be found in the geophysical
processing of seismic data where the effect of the source response
(i.e., airgun or vibroseis) should be removed from the recorded signal
to be able to better interpret the response of the subsurface. Similar examples
can be found in telecommunication and speech analysis.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.sparse.linalg import lsqr

import pylops
from pylops.utils.wavelets import ricker

plt.close("all")

###############################################################################
# We will start by creating a zero signal of lenght :math:`nt` and we will
# place a unitary spike at its center. We also create our filter to be
# applied by means of :py:class:`pylops.signalprocessing.Convolve1D` operator.
# Following the seismic example mentioned above, the filter is a
# `Ricker wavelet <http://subsurfwiki.org/wiki/Ricker_wavelet>`_
# with dominant frequency :math:`f_0 = 30 Hz`.
nt = 1001
dt = 0.004
t = np.arange(nt) * dt
x = np.zeros(nt)
x[int(nt / 2)] = 1
h, th, hcenter = ricker(t[:101], f0=30)

Cop = pylops.signalprocessing.Convolve1D(nt, h=h, offset=hcenter, dtype="float32")
y = Cop * x

xinv = Cop / y

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x, "k", lw=2, label=r"$x$")
ax.plot(t, y, "r", lw=2, label=r"$y=Ax$")
ax.plot(t, xinv, "--g", lw=2, label=r"$x_{ext}$")
ax.set_title("Convolve 1d data", fontsize=14, fontweight="bold")
ax.legend()
ax.set_xlim(1.9, 2.1)
plt.tight_layout()

###############################################################################
# We show now that also a filter with mixed phase (i.e., not centered
# around zero) can be applied and inverted for using the
# :py:class:`pylops.signalprocessing.Convolve1D`
# operator.
Cop = pylops.signalprocessing.Convolve1D(nt, h=h, offset=hcenter - 3, dtype="float32")
y = Cop * x
y1 = Cop.H * x
xinv = Cop / y

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x, "k", lw=2, label=r"$x$")
ax.plot(t, y, "r", lw=2, label=r"$y=Ax$")
ax.plot(t, y1, "b", lw=2, label=r"$y=A^Hx$")
ax.plot(t, xinv, "--g", lw=2, label=r"$x_{ext}$")
ax.set_title(
    "Convolve 1d data with non-zero phase filter", fontsize=14, fontweight="bold"
)
ax.set_xlim(1.9, 2.1)
ax.legend()
plt.tight_layout()

###############################################################################
# We repeat a similar exercise but using two dimensional signals and
# filters taking advantage of the
# :py:class:`pylops.signalprocessing.Convolve2D` operator.
nt = 51
nx = 81
dt = 0.004
t = np.arange(nt) * dt
x = np.zeros((nt, nx))
x[int(nt / 2), int(nx / 2)] = 1

nh = [11, 5]
h = np.ones((nh[0], nh[1]))

Cop = pylops.signalprocessing.Convolve2D(
    (nt, nx),
    h=h,
    offset=(int(nh[0]) / 2, int(nh[1]) / 2),
    dtype="float32",
)
y = Cop * x
xinv = (Cop / y.ravel()).reshape(Cop.dims)

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
fig.suptitle("Convolve 2d data", fontsize=14, fontweight="bold", y=0.95)
axs[0].imshow(x, cmap="gray", vmin=-1, vmax=1)
axs[1].imshow(y, cmap="gray", vmin=-1, vmax=1)
axs[2].imshow(xinv, cmap="gray", vmin=-1, vmax=1)
axs[0].set_title("x")
axs[0].axis("tight")
axs[1].set_title("y")
axs[1].axis("tight")
axs[2].set_title("xlsqr")
axs[2].axis("tight")
plt.tight_layout()
plt.subplots_adjust(top=0.8)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle("Convolve in 2d data - traces", fontsize=14, fontweight="bold", y=0.95)
ax[0].plot(x[int(nt / 2), :], "k", lw=2, label=r"$x$")
ax[0].plot(y[int(nt / 2), :], "r", lw=2, label=r"$y=Ax$")
ax[0].plot(xinv[int(nt / 2), :], "--g", lw=2, label=r"$x_{ext}$")
ax[1].plot(x[:, int(nx / 2)], "k", lw=2, label=r"$x$")
ax[1].plot(y[:, int(nx / 2)], "r", lw=2, label=r"$y=Ax$")
ax[1].plot(xinv[:, int(nx / 2)], "--g", lw=2, label=r"$x_{ext}$")
ax[0].legend()
ax[0].set_xlim(30, 50)
ax[1].legend()
ax[1].set_xlim(10, 40)
plt.tight_layout()
plt.subplots_adjust(top=0.8)

###############################################################################
# We now do the same using three dimensional signals and
# filters taking advantage of the
# :py:class:`pylops.signalprocessing.ConvolveND` operator.
ny, nx, nz = 13, 10, 7
x = np.zeros((ny, nx, nz))
x[ny // 3, nx // 2, nz // 4] = 1
h = np.ones((3, 5, 3))
offset = [1, 2, 1]

Cop = pylops.signalprocessing.ConvolveND(
    dims=(ny, nx, nz), h=h, offset=offset, axes=(0, 1, 2), dtype="float32"
)
y = Cop * x
xlsqr = lsqr(Cop, y.ravel(), damp=0, iter_lim=300, show=0)[0]
xlsqr = xlsqr.reshape(Cop.dims)

fig, axs = plt.subplots(3, 3, figsize=(10, 12))
fig.suptitle("Convolve 3d data", y=0.98, fontsize=14, fontweight="bold")
axs[0][0].imshow(x[ny // 3], cmap="gray", vmin=-1, vmax=1)
axs[0][1].imshow(y[ny // 3], cmap="gray", vmin=-1, vmax=1)
axs[0][2].imshow(xlsqr[ny // 3], cmap="gray", vmin=-1, vmax=1)
axs[0][0].set_title("x")
axs[0][0].axis("tight")
axs[0][1].set_title("y")
axs[0][1].axis("tight")
axs[0][2].set_title("xlsqr")
axs[0][2].axis("tight")
axs[1][0].imshow(x[:, nx // 2], cmap="gray", vmin=-1, vmax=1)
axs[1][1].imshow(y[:, nx // 2], cmap="gray", vmin=-1, vmax=1)
axs[1][2].imshow(xlsqr[:, nx // 2], cmap="gray", vmin=-1, vmax=1)
axs[1][0].axis("tight")
axs[1][1].axis("tight")
axs[1][2].axis("tight")
axs[2][0].imshow(x[..., nz // 4], cmap="gray", vmin=-1, vmax=1)
axs[2][1].imshow(y[..., nz // 4], cmap="gray", vmin=-1, vmax=1)
axs[2][2].imshow(xlsqr[..., nz // 4], cmap="gray", vmin=-1, vmax=1)
axs[2][0].axis("tight")
axs[2][1].axis("tight")
axs[2][2].axis("tight")
plt.tight_layout()

###############################################################################
# Up until now, we have only considered the case where the filter is compact
# and much shorter of the input data. There are however scenarios where the
# opposite (i.e., filter is longer than the signal) is desired. This is for
# example the case when one wants to estimate a filter (:math:`\mathbf{h}`)
# to match two signals (:math:`\mathbf{x}` and :math:`\mathbf{y}`):
#
# .. math::
#   J = || \mathbf{y} - \mathbf{X} \mathbf{h} ||_2^2
#
# Such a scenario is very commonly used in so-called adaptive substraction
# techniques. We will try now to use :py:class:`pylops.signalprocessing.Convolve1D`
# to match two signals that have both a phase and amplitude mismatch.

# Define input signal
nt = 101
dt = 0.004
t = np.arange(nt) * dt

x = np.zeros(nt)
x[[int(nt / 4), int(nt / 2), int(2 * nt / 3)]] = [3, -2, 1]
h, th, hcenter = ricker(t[:41], f0=20)

Cop = pylops.signalprocessing.Convolve1D(nt, h=h, offset=hcenter, dtype="float32")
x = Cop * x

# Phase and amplitude corrupt the input
amp = 0.9
phase = 40

y = amp * (
    x * np.cos(np.deg2rad(phase))
    + np.imag(sp.signal.hilbert(x)) * np.sin(np.deg2rad(phase))
)

# Define convolution operator
nh = 21
th = np.arange(nh) * dt - dt * nh // 2
Yop = pylops.signalprocessing.Convolve1D(nh, h=y, offset=nh // 2)

# Find filter to match x to y
h = Yop / x
ymatched = Yop @ h

# Find sparse filter to match x to y
hsparse = pylops.optimization.sparsity.fista(Yop, x, niter=100, eps=1e-1)[0]
ymatchedsparse = Yop @ hsparse

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x, "k", lw=2, label=r"$x$")
ax.plot(t, y, "r", lw=2, label=r"$y$")
ax.plot(t, ymatched, "--g", lw=2, label=r"$y_{matched}$")
ax.plot(t, x - ymatched, "--k", lw=2, label=r"$x-y_{matched,sparse}$")
ax.plot(t, ymatchedsparse, "--m", lw=2, label=r"$y_{matched}$")
ax.plot(t, x - ymatchedsparse, "--k", lw=2, label=r"$x-y_{matched,sparse}$")
ax.set_title("Signals to match", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
plt.tight_layout()

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].plot(th, h, "k", lw=2)
axs[0].set_title("Matching filter", fontsize=14, fontweight="bold")
axs[1].plot(th, hsparse, "k", lw=2)
axs[1].set_title("Sparse Matching filter", fontsize=14, fontweight="bold")
plt.tight_layout()


###############################################################################
# Finally, in some cases one wants to convolve (or correlate) two signals of
# the same size. This can also be obtained using
# :py:class:`pylops.signalprocessing.Convolve1D`. We will see here a case
# where the operator is used to trace-wise auto-correlate signals from
# a 2-dimensional array representing a seismic dataset.

# Create data
par = {"ox": -140, "dx": 2, "nx": 140, "ot": 0, "dt": 0.004, "nt": 200, "f0": 20}

v = 1500
t0 = [0.2, 0.4, 0.5]
px = [0, 0, 0]
pxx = [1e-5, 5e-6, 1e-20]
amp = [1.0, -2, 0.5]

t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)
wav = pylops.utils.wavelets.ricker(t[:41], f0=par["f0"])[0]
_, data = pylops.utils.seismicevents.parabolic2d(x, t, t0, px, pxx, amp, wav)

# Convolution operator
Xop = pylops.signalprocessing.Convolve1D(
    (par["nx"], par["nt"]), h=data, offset=par["nt"] // 2, axis=-1, method="fft"
)

corr = Xop.H @ data

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].imshow(data.T, cmap="gray", vmin=-1, vmax=1, extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set_xlabel("Rec (m)", fontsize=14, fontweight="bold")
axs[0].set_ylabel("T (s)", fontsize=14, fontweight="bold")
axs[0].set_title("Data", fontsize=14, fontweight="bold")
axs[0].axis("tight")
axs[1].imshow(
    corr.T,
    cmap="gray",
    vmin=-10,
    vmax=10,
    extent=(x[0], x[-1], t[par["nt"] // 2], -t[par["nt"] // 2]),
)
axs[1].set_xlabel("Rec (m)", fontsize=14, fontweight="bold")
axs[1].set_title("Auto-correlation", fontsize=14, fontweight="bold")
axs[1].axis("tight")
plt.tight_layout()
