r"""
Fourier Radon Transform
=======================
This example shows how to use the :py:class:`pylops.signalprocessing.FourierRadon2D`
operator to apply the linear and parabolic Radon Transform to 2-dimensional signals.

This operator provides a transformation equivalent to that of
:py:class:`pylops.signalprocessing.Radon2D`, however since the shift-and-sum step
is performed in the frequency domain, this is analytically correct (compared to
performing to shifting the data via nearest or linear interpolation).

"""
import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

###############################################################################
# Let's start by creating a empty 2d matrix of size :math:`n_x \times n_t`
# with a single linear event.

par = {
    "ot": 0,
    "dt": 0.004,
    "nt": 51,
    "ox": -250,
    "dx": 10,
    "nx": 51,
    "oy": -250,
    "dy": 10,
    "ny": 51,
    "f0": 40,
}
theta = [10.0]
t0 = [0.1]
amp = [1.0]

# Create axes
t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)
dt, dx, dy = par["dt"], par["dx"], par["dy"]

# Create wavelet
wav, _, wav_c = pylops.utils.wavelets.ricker(t[:41], f0=par["f0"])

# Generate data
_, d = pylops.utils.seismicevents.linear2d(x, t, 1500.0, t0, theta, amp, wav)


###############################################################################
# We can now define our operators and apply the forward, adjoint and inverse
# steps.
nfft = int(2 ** np.ceil(np.log2(par["nt"])))
npx, pxmax = 2 * par["nx"], 5e-4
px = np.linspace(-pxmax, pxmax, npx)

R2Op = pylops.signalprocessing.FourierRadon2D(
    t, x, px, nfft, kind="linear", engine="numpy", dtype="float64"
)
dL_chirp = R2Op.H * d
dadj_chirp = R2Op * dL_chirp

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(d.T, vmin=-1, vmax=1, cmap="bwr_r", extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input linear")
axs[0].axis("tight")
axs[1].imshow(
    dL_chirp.T,
    cmap="bwr_r",
    vmin=-dL_chirp.max(),
    vmax=dL_chirp.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * np.sin(np.deg2rad(theta[0])) / 1500.0, t0[0], s=50, color="r")
axs[1].set(xlabel=r"$p$ [s/km]", title="Radon")
axs[1].axis("tight")
axs[2].imshow(
    dadj_chirp.T,
    cmap="bwr_r",
    vmin=-dadj_chirp.max(),
    vmax=dadj_chirp.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$x$ [m]", title="Adj Radon")
axs[2].axis("tight")
plt.tight_layout()


###############################################################################
# We repeat now the same with a parabolic event

# Generate data
pxx = [1e-6]
_, d = pylops.utils.seismicevents.parabolic2d(x, t, t0, 0, np.array(pxx), amp, wav)

# Radon transform
npx, pxmax = 2 * par["nx"], 5e-6
px = np.linspace(-pxmax, pxmax, npx)

R2Op = pylops.signalprocessing.FourierRadon2D(
    t, x, px, nfft, kind="parabolic", engine="numpy", dtype="float64"
)
dL_chirp = R2Op.H * d
dadj_chirp = R2Op * dL_chirp

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(d.T, vmin=-1, vmax=1, cmap="bwr_r", extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input parabolic")
axs[0].axis("tight")
axs[1].imshow(
    dL_chirp.T,
    cmap="bwr_r",
    vmin=-dL_chirp.max(),
    vmax=dL_chirp.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pxx[0], t0[0], s=50, color="r")
axs[1].set(xlabel=r"$p$ [s/km]", title="Radon")
axs[1].axis("tight")
axs[2].imshow(
    dadj_chirp.T,
    cmap="bwr_r",
    vmin=-dadj_chirp.max(),
    vmax=dadj_chirp.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$x$ [m]", title="Adj Radon")
axs[2].axis("tight")
plt.tight_layout()
