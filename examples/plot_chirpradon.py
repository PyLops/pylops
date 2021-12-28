r"""
Chirp Radon Transform
=====================
This example shows how to use the :py:class:`pylops.signalprocessing.ChirpRadon2D`
and :py:class:`pylops.signalprocessing.ChirpRadon3D` operators to apply the
linear Radon Transform to 2-dimensional or 3-dimensional signals, respectively.

When working with the linear Radon transform, this is a faster implementation
compared to in :py:class:`pylops.signalprocessing.Radon2D` and
:py:class:`pylops.signalprocessing.Radon3D` and should be preferred.
This method provides also an analytical inverse.

Note that the forward and adjoint definitions in these two pairs of operators
are swapped.

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
theta = [
    0,
]
t0 = [
    0.1,
]
amp = [
    1.0,
]

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
npx, pxmax = par["nx"], 5e-4
px = np.linspace(-pxmax, pxmax, npx)

R2Op = pylops.signalprocessing.ChirpRadon2D(t, x, pxmax * dx / dt, dtype="float64")
dL_chirp = R2Op * d.ravel()
dadj_chirp = R2Op.H * dL_chirp
dinv_chirp = R2Op.inverse(dL_chirp)

dL_chirp = dL_chirp.reshape(par["nx"], par["nt"])
dadj_chirp = dadj_chirp.reshape(par["nx"], par["nt"])
dinv_chirp = dinv_chirp.reshape(par["nx"], par["nt"])

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(d.T, vmin=-1, vmax=1, cmap="seismic_r", extent=(x[0], x[-1], t[-1], t[0]))
axs[0].set_title("Input model")
axs[0].axis("tight")
axs[1].imshow(
    dL_chirp.T,
    cmap="seismic_r",
    vmin=-dL_chirp.max(),
    vmax=dL_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[1].set_title("Radon Chirp")
axs[1].axis("tight")
axs[2].imshow(
    dadj_chirp.T,
    cmap="seismic_r",
    vmin=-dadj_chirp.max(),
    vmax=dadj_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[2].set_title("Adj Radon Chirp")
axs[2].axis("tight")
axs[3].imshow(
    dinv_chirp.T,
    cmap="seismic_r",
    vmin=-d.max(),
    vmax=d.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[3].set_title("Inv Radon Chirp")
axs[3].axis("tight")
plt.tight_layout()


###############################################################################
# Finally we repeat the same exercise with 3d data.

par = {
    "ot": 0,
    "dt": 0.004,
    "nt": 51,
    "ox": -400,
    "dx": 10,
    "nx": 81,
    "oy": -600,
    "dy": 10,
    "ny": 61,
    "f0": 20,
}
theta = [
    10,
]
phi = [
    0,
]
t0 = [
    0.1,
]
amp = [
    1.0,
]

# Create axes
t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)
dt, dx, dy = par["dt"], par["dx"], par["dy"]

# Generate data
_, d = pylops.utils.seismicevents.linear3d(x, y, t, 1500.0, t0, theta, phi, amp, wav)

npy, pymax = par["ny"], 3e-4
npx, pxmax = par["nx"], 5e-4
py = np.linspace(-pymax, pymax, npy)
px = np.linspace(-pxmax, pxmax, npx)

R3Op = pylops.signalprocessing.ChirpRadon3D(
    t, y, x, (pymax * dy / dt, pxmax * dx / dt), dtype="float64"
)
dL_chirp = R3Op * d.ravel()
dadj_chirp = R3Op.H * dL_chirp
dinv_chirp = R3Op.inverse(dL_chirp)

dL_chirp = dL_chirp.reshape(par["ny"], par["nx"], par["nt"])
dadj_chirp = dadj_chirp.reshape(par["ny"], par["nx"], par["nt"])
dinv_chirp = dinv_chirp.reshape(par["ny"], par["nx"], par["nt"])


fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(
    d[par["ny"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="seismic_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set_title("Input model")
axs[0].axis("tight")
axs[1].imshow(
    dL_chirp[par["ny"] // 2].T,
    cmap="seismic_r",
    vmin=-dL_chirp.max(),
    vmax=dL_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[1].set_title("Radon Chirp")
axs[1].axis("tight")
axs[2].imshow(
    dadj_chirp[par["ny"] // 2].T,
    cmap="seismic_r",
    vmin=-dadj_chirp.max(),
    vmax=dadj_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[2].set_title("Adj Radon Chirp")
axs[2].axis("tight")
axs[3].imshow(
    dinv_chirp[par["ny"] // 2].T,
    cmap="seismic_r",
    vmin=-d.max(),
    vmax=d.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[3].set_title("Inv Radon Chirp")
axs[3].axis("tight")
plt.tight_layout()

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(
    d[:, par["nx"] // 2].T,
    vmin=-1,
    vmax=1,
    cmap="seismic_r",
    extent=(x[0], x[-1], t[-1], t[0]),
)
axs[0].set_title("Input model")
axs[0].axis("tight")
axs[1].imshow(
    dL_chirp[:, 2 * par["nx"] // 3].T,
    cmap="seismic_r",
    vmin=-dL_chirp.max(),
    vmax=dL_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[1].set_title("Radon Chirp")
axs[1].axis("tight")
axs[2].imshow(
    dadj_chirp[:, par["nx"] // 2].T,
    cmap="seismic_r",
    vmin=-dadj_chirp.max(),
    vmax=dadj_chirp.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[2].set_title("Adj Radon Chirp")
axs[2].axis("tight")
axs[3].imshow(
    dinv_chirp[:, par["nx"] // 2].T,
    cmap="seismic_r",
    vmin=-d.max(),
    vmax=d.max(),
    extent=(px[0], px[-1], t[-1], t[0]),
)
axs[3].set_title("Inv Radon Chirp")
axs[3].axis("tight")
plt.tight_layout()
