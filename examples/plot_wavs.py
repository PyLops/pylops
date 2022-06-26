"""
Wavelets
========
This example shows how to use the different wavelets available PyLops.
"""
import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

###############################################################################
# Let's start with defining a time axis and creating the FFT operator
dt = 0.004
nt = 1001
t = np.arange(nt) * dt

Fop = pylops.signalprocessing.FFT(2 * nt - 1, sampling=dt, real=True)
f = Fop.f

###############################################################################
# We can now create the different wavelets and display them

# Gaussian
wg, twg, wgc = pylops.utils.wavelets.gaussian(t, std=2)

# Gaussian
wk, twk, wgk = pylops.utils.wavelets.klauder(t, f=[4, 30], taper=np.hanning)

# Ormsby
wo, two, woc = pylops.utils.wavelets.ormsby(t, f=[5, 9, 25, 30], taper=np.hanning)

# Ricker
wr, twr, wrc = pylops.utils.wavelets.ricker(t, f0=17)

# Frequency domain
wgf = Fop @ wg
wkf = Fop @ wk
wof = Fop @ wo
wrf = Fop @ wr

###############################################################################
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].plot(twg, wg, "k", lw=2, label="Gaussian")
axs[0].plot(twk, wk, "r", lw=2, label="Klauder")
axs[0].plot(two, wo, "b", lw=2, label="Ormsby")
axs[0].plot(twr, wr, "y--", lw=2, label="Ricker")
axs[0].set(xlim=(-0.4, 0.4), xlabel="Time [s]")
axs[0].legend()
axs[1].plot(f, np.abs(wgf) / np.abs(wgf).max(), "k", lw=2, label="Gaussian")
axs[1].plot(f, np.abs(wkf) / np.abs(wkf).max(), "r", lw=2, label="Klauder")
axs[1].plot(f, np.abs(wof) / np.abs(wof).max(), "b", lw=2, label="Ormsby")
axs[1].plot(f, np.abs(wrf) / np.abs(wrf).max(), "y--", lw=2, label="Ricker")
axs[1].set(xlim=(0, 50), xlabel="Frequency [Hz]")
axs[1].legend()
plt.tight_layout()
