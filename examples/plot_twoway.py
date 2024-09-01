r"""
Acoustic Wave Equation modelling
================================

This example shows how to perform acoustic wave equation modelling
using the :class:`pylops.waveeqprocessing.AcousticWave2D` operator,
which brings the power of finite-difference modelling via the Devito
modelling engine to PyLops.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import pylops

plt.close("all")
np.random.seed(0)


###############################################################################
# To begin with, we will create a simple layered velocity model. We will also
# define a background velocity model by smoothing the original velocity model
# which will be responsible of the kinematic of the wavefield modelled via
# Born modelling, and the perturbation velocity model which will lead to
# scattering effects and therefore guide the dynamic of the modelled wavefield.

# Velocity Model
nx, nz = 61, 40
dx, dz = 4, 4
x, z = np.arange(nx) * dx, np.arange(nz) * dz
vel = 1000 * np.ones((nx, nz))
vel[:, 15:] = 1200
vel[:, 35:] = 1600

# Smooth velocity model
v0 = gaussian_filter(vel, sigma=10)

# Born perturbation from m - m0
dv = vel ** (-2) - v0 ** (-2)

# Receivers
nr = 101
rx = np.linspace(0, x[-1], nr)
rz = 20 * np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0, 1] - recs[0, 0]

# Sources
ns = 3
sx = np.linspace(0, x[-1], ns)
sz = 10 * np.ones(ns)
sources = np.vstack((sx, sz))

plt.figure(figsize=(10, 5))
im = plt.imshow(vel.T, cmap="summer", extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0], recs[1], marker="v", s=150, c="b", edgecolors="k")
plt.scatter(sources[0], sources[1], marker="*", s=150, c="r", edgecolors="k")
cb = plt.colorbar(im)
cb.set_label("[m/s]")
plt.axis("tight")
plt.xlabel("x [m]"), plt.ylabel("z [m]")
plt.title("Velocity")
plt.xlim(x[0], x[-1])
plt.tight_layout()

plt.figure(figsize=(10, 5))
im = plt.imshow(dv.T, cmap="seismic", extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0], recs[1], marker="v", s=150, c="b", edgecolors="k")
plt.scatter(sources[0], sources[1], marker="*", s=150, c="r", edgecolors="k")
cb = plt.colorbar(im)
cb.set_label("[m/s]")
plt.axis("tight")
plt.xlabel("x [m]"), plt.ylabel("z [m]")
plt.title("Velocity perturbation")
plt.xlim(x[0], x[-1])
plt.tight_layout()

###############################################################################
# Let us now define the Born modelling operator

Aop = pylops.waveeqprocessing.AcousticWave2D(
    (nx, nz),
    (0, 0),
    (dx, dz),
    v0,
    sources[0],
    sources[1],
    recs[0],
    recs[1],
    0.0,
    0.5 * 1e3,
    "Ricker",
    space_order=4,
    nbl=100,
    f0=15,
    dtype="float32",
)

###############################################################################
# And we use it to model our data

dobs = Aop @ dv

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 6))
fig.suptitle("FD modelling with Ricker", y=0.99)

for isrc in range(ns):
    axs[isrc].imshow(
        dobs[isrc].reshape(Aop.geometry.nrec, Aop.geometry.nt).T,
        cmap="gray",
        vmin=-1e-7,
        vmax=1e-7,
        extent=(
            recs[0, 0],
            recs[0, -1],
            Aop.geometry.time_axis.time_values[-1] * 1e-3,
            0,
        ),
    )
    axs[isrc].axis("tight")
    axs[isrc].set_xlabel("rec [m]")
axs[0].set_ylabel("t [s]")
fig.tight_layout()

###############################################################################
# Finally, we are going to show how despite the
# :class:`pylops.waveeqprocessing.AcousticWave2D` operator allows a user to
# specify a limited number of source wavelets (this is directly borrowed from
# Devito), a simple modification can be applied to pass any user defined wavelet.
# We are going to do that with a Ormsby wavelet

# Extract Ricker wavelet
wav = Aop.geometry.src.data[:, 0]
wavc = np.argmax(wav)

# Define Ormsby wavelet
wavest = pylops.utils.wavelets.ormsby(
    Aop.geometry.time_axis.time_values[:wavc] * 1e-3, f=[3, 20, 30, 45]
)[0]

# Update wavelet in operator and model new data
Aop.updatesrc(wavest)

dobs1 = Aop @ dv

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 6))
fig.suptitle("FD modelling with Ormsby", y=0.99)

for isrc in range(ns):
    axs[isrc].imshow(
        dobs1[isrc].reshape(Aop.geometry.nrec, Aop.geometry.nt).T,
        cmap="gray",
        vmin=-1e-7,
        vmax=1e-7,
        extent=(
            recs[0, 0],
            recs[0, -1],
            Aop.geometry.time_axis.time_values[-1] * 1e-3,
            0,
        ),
    )
    axs[isrc].axis("tight")
    axs[isrc].set_xlabel("rec [m]")
axs[0].set_ylabel("t [s]")
fig.tight_layout()

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].plot(wav[: 2 * wavc], "k")
axs[0].plot(wavest, "r")
axs[1].plot(
    dobs[isrc].reshape(Aop.geometry.nrec, Aop.geometry.nt)[nr // 2], "k", label="Ricker"
)
axs[1].plot(
    dobs1[isrc].reshape(Aop.geometry.nrec, Aop.geometry.nt)[nr // 2],
    "r",
    label="Ormsby",
)
axs[1].legend()
fig.tight_layout()
