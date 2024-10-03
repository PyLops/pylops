r"""
Fourier Radon Transform
=======================
This example shows how to use the :py:class:`pylops.signalprocessing.FourierRadon2D`
and :py:class:`pylops.signalprocessing.FourierRadon3D` operators to apply the linear
and parabolic Radon Transform to 2-dimensional or 3-dimensional signals, respectively.

These operators provides transformations equivalent to those of
:py:class:`pylops.signalprocessing.Radon2D` and :py:class:`pylops.signalprocessing.Radon3D`,
however since the shift-and-sum step is performed in the frequency domain,
this is analytically correct (compared to performing to shifting the data via
nearest or linear interpolation).

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
# We can now define our operators and apply the forward and adjoint
# steps.
nfft = int(2 ** np.ceil(np.log2(par["nt"])))
npx, pxmax = 2 * par["nx"], 5e-4
px = np.linspace(-pxmax, pxmax, npx)

R2Op = pylops.signalprocessing.FourierRadon2D(
    t, x, px, nfft, kind="linear", engine="numpy", dtype="float64"
)
dL = R2Op.H * d
dadj = R2Op * dL

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(d.T, vmin=-1, vmax=1, cmap="bwr_r", extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input linear")
axs[0].axis("tight")
axs[1].imshow(
    dL.T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * np.sin(np.deg2rad(theta[0])) / 1500.0, t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p$ [s/km]", title="Radon")
axs[1].axis("tight")
axs[2].imshow(
    dadj.T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
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
dL = R2Op.H * d
dadj = R2Op * dL

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(d.T, vmin=-1, vmax=1, cmap="bwr_r", extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input parabolic")
axs[0].axis("tight")
axs[1].imshow(
    dL.T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pxx[0], t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p$ [s/km]", title="Radon")
axs[1].axis("tight")
axs[2].imshow(
    dadj.T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$x$ [m]", title="Adj Radon")
axs[2].axis("tight")
plt.tight_layout()

###############################################################################
# Finally we repeat the same exercise with 3d data.

par = {
    "ot": 0,
    "dt": 0.004,
    "nt": 51,
    "ox": -100,
    "dx": 10,
    "nx": 21,
    "oy": -200,
    "dy": 10,
    "ny": 41,
    "f0": 20,
}
theta = [30]
phi = [10]
t0 = [0.1]
amp = [1.0]

# Create axes
t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)
dt, dx, dy = par["dt"], par["dx"], par["dy"]

# Generate linear data
pxx = np.sin(np.deg2rad(theta[0])) * np.cos(np.deg2rad(phi[0])) / 1500.0
pyy = np.sin(np.deg2rad(theta[0])) * np.sin(np.deg2rad(phi[0])) / 1500.0
_, d = pylops.utils.seismicevents.linear3d(x, y, t, 1500.0, t0, theta, phi, amp, wav)

# Linear Radon
npy, pymax = par["ny"], 5e-4
npx, pxmax = par["nx"], 5e-4
py = np.linspace(-pymax, pymax, npy)
px = np.linspace(-pxmax, pxmax, npx)

R3Op = pylops.signalprocessing.FourierRadon3D(
    t, y, x, py, px, nfft, kind=("linear", "linear"), engine="numpy", dtype="float64"
)
dL = R3Op.H * d
dadj = R3Op * dL

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(
    d[par["ny"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="bwr_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input linear 3d - y")
axs[0].axis("tight")
axs[1].imshow(
    dL[np.argmin(np.abs(pyy - py))].T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pxx, t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p_x$ [s/km]", title="Radon 3d - y")
axs[1].axis("tight")
axs[2].imshow(
    dadj[par["ny"] // 2].T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$x$ [m]", title="Adj Radon 3d - y")
axs[2].axis("tight")
plt.tight_layout()

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(
    d[:, par["nx"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="bwr_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set(xlabel=r"$y$ [m]", ylabel=r"$t$ [s]", title="Input linear 3d - x")
axs[0].axis("tight")
axs[1].imshow(
    dL[:, np.argmin(np.abs(pxx - px))].T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * py[0], 1e3 * py[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pyy, t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p_y$ [s/km]", title="Radon 3d - x")
axs[1].axis("tight")
axs[2].imshow(
    dadj[:, par["nx"] // 2].T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$y$ [m]", title="Adj Radon 3d - x")
axs[2].axis("tight")
plt.tight_layout()

# Generate parabolic data
pxx = [1e-6]
pyy = [2e-6]
_, d = pylops.utils.seismicevents.parabolic3d(
    x, y, t, t0, 0, 0, np.array(pxx), np.array(pyy), amp, wav
)

# Parabolic Radon
npy, pymax = par["ny"], 5e-6
npx, pxmax = par["nx"], 5e-6
py = np.linspace(-pymax, pymax, npy)
px = np.linspace(-pxmax, pxmax, npx)

R3Op = pylops.signalprocessing.FourierRadon3D(
    t,
    y,
    x,
    py,
    px,
    nfft,
    kind=("parabolic", "parabolic"),
    engine="numpy",
    dtype="float64",
)
dL = R3Op.H * d
dadj = R3Op * dL

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(
    d[par["ny"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="bwr_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set(xlabel=r"$x$ [m]", ylabel=r"$t$ [s]", title="Input parabolic 3d - y")
axs[0].axis("tight")
axs[1].imshow(
    dL[np.argmin(np.abs(pyy - py))].T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * px[0], 1e3 * px[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pxx[0], t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p_x$ [s/km]", title="Radon 3d - y")
axs[1].axis("tight")
axs[2].imshow(
    dadj[par["ny"] // 2].T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$x$ [m]", title="Adj Radon 3d - y")
axs[2].axis("tight")
plt.tight_layout()

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axs[0].imshow(
    d[:, par["nx"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="bwr_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set(xlabel=r"$y$ [m]", ylabel=r"$t$ [s]", title="Input parabolic 3d - x")
axs[0].axis("tight")
axs[1].imshow(
    dL[:, np.argmin(np.abs(pxx - px))].T,
    cmap="bwr_r",
    vmin=-dL.max(),
    vmax=dL.max(),
    extent=(1e3 * py[0], 1e3 * py[-1], t[-1], t[0]),
)
axs[1].scatter(1e3 * pyy[0], t0[0], s=50, color="k")
axs[1].set(xlabel=r"$p_y$ [s/km]", title="Radon 3d - x")
axs[1].axis("tight")
axs[2].imshow(
    dadj[:, par["nx"] // 2].T,
    cmap="bwr_r",
    vmin=-dadj.max(),
    vmax=dadj.max(),
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[2].set(xlabel=r"$y$ [m]", title="Adj Radon 3d - x")
axs[2].axis("tight")
plt.tight_layout()
