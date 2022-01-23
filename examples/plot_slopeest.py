r"""
Slope estimation via Structure Tensor algorithm
===============================================

This example shows how to estimate local slopes of a two-dimensional array
using :py:func:`pylops.utils.signalprocessing.slope_estimate`.

Knowing the local slopes of an image (or a seismic data) can be useful for
a variety of tasks in image (or geophysical) processing such as denoising,
smoothing, or interpolation. When slopes are used with the
:py:class:`pylops.signalprocessing.Seislet` operator, the input dataset can be
compressed and the sparse nature of the Seislet transform can also be used to
precondition sparsity-promoting inverse problems.

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pylops
from pylops.signalprocessing.Seislet import _predict_trace

plt.close("all")
np.random.seed(10)

###############################################################################
# To start we import a 2d image and estimate the local slopes of the image.
im = np.load("../testdata/python.npy")[..., 0]
im = im / 255.0 - 0.5

slopes, anisotropy = pylops.utils.signalprocessing.slope_estimate(im, smooth=7)
angles = -np.rad2deg(np.arctan(slopes))

###############################################################################
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
iax = axs[0].imshow(im, cmap="viridis", origin="lower")
axs[0].set_title("Data")
cax = make_axes_locatable(axs[0]).append_axes("right", size="5%", pad=0.05)
cax.axis("off")

iax = axs[1].imshow(angles, cmap="RdBu_r", origin="lower", vmin=-90, vmax=90)
axs[1].set_title("Angle of incline [°]")
cax = make_axes_locatable(axs[1]).append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(iax, cax=cax, orientation="vertical")

iax = axs[2].imshow(anisotropy, cmap="Reds", origin="lower", vmin=0, vmax=1)
axs[2].set_title("Anisotropy")
cax = make_axes_locatable(axs[2]).append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(iax, cax=cax, orientation="vertical")
fig.tight_layout()

###############################################################################
# We can now repeat the same using some seismic data. We will first define
# a single trace and a slope field, apply such slope field to the trace
# recursively to create the other traces of the data and finally try to recover
# the underlying slope field from the data alone.

# Reflectivity model
nx, nt = 2 ** 7, 121
dx, dt = 0.01, 0.004
x, t = np.arange(nx) * dx, np.arange(nt) * dt

nspike = nt // 8
refl = np.zeros(nt)
it = np.sort(np.random.permutation(range(10, nt - 20))[:nspike])
refl[it] = np.random.normal(0.0, 1.0, nspike)

# Wavelet
ntwav = 41
f0 = 30
twav = np.arange(ntwav) * dt
wav, *_ = pylops.utils.wavelets.ricker(twav, f0)

# Input trace
trace = np.convolve(refl, wav, mode="same")

# Slopes
theta = np.deg2rad(np.linspace(0, 30, nx))
slope = np.outer(np.ones(nt), np.tan(theta) * dt / dx)

# Model data
d = np.zeros((nt, nx))
tr = trace.copy()
for ix in range(nx):
    tr = _predict_trace(tr, t, dt, dx, slope[:, ix])
    d[:, ix] = tr

# Estimate slopes
slope_est, _ = pylops.utils.signalprocessing.slope_estimate(d, dt, dx, smooth=10)
slope_est *= -1

###############################################################################
fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)

opts = dict(aspect="auto", extent=(x[0], x[-1], t[-1], t[0]))
iax = axs[0, 0].imshow(d, cmap="gray", vmin=-1, vmax=1, **opts)
axs[0, 0].set(title="Data", ylabel="Time [s]")
cax = make_axes_locatable(axs[0, 0]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(iax, cax=cax, orientation="vertical")

opts.update(dict(cmap="RdBu_r", vmin=np.min(slope), vmax=np.max(slope)))
iax = axs[0, 1].imshow(slope, **opts)
axs[0, 1].set(title="True Slope")
cax = make_axes_locatable(axs[0, 1]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(iax, cax=cax, orientation="vertical")
cax.set_ylabel("[s/km]")

iax = axs[1, 0].imshow(np.abs(slope - slope_est), **opts)
axs[1, 0].set(
    title="Estimate absolute error", ylabel="Time [s]", xlabel="Position [km]"
)
cax = make_axes_locatable(axs[1, 0]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(iax, cax=cax, orientation="vertical")
cax.set_ylabel("[s/km]")

iax = axs[1, 1].imshow(slope_est, **opts)
axs[1, 1].set(title="Estimated Slope", xlabel="Position [km]")
cax = make_axes_locatable(axs[1, 1]).append_axes("right", size="5%", pad=0.05)
fig.colorbar(iax, cax=cax, orientation="vertical")
cax.set_ylabel("[s/km]")

fig.tight_layout()

###############################################################################
# As you can see the Structure Tensor algorithm is a very fast, general purpose
# algorithm that can be used to estimate local slopes to input datasets of
# very different nature.
