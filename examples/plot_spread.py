r"""
How to Spreading
================
This example focuses on the :py:class:`pylops.basicoperators.Spread` operator,
which is a highly versatile operator in PyLops to perform spreading/stacking
operations in a vectorized manner (or efficiently via numba-jited for loops).

Whilst :py:class:`pylops.basicoperators.Spread` is powerful, it may not be
obvious at first how to leverage it properly. Whilst looking at the
:py:class:`pylops.signalprocessing.Radon2D` and
:py:class:`pylops.signalprocessing.Radon3D` operators is highly reccomended as
they are built using directly the :py:class:`pylops.basicoperators.Spread` operator,
in the example we will recreate a simplified version of the famous linear
Radon operator, which stacks data along straigh lines with a given intercept
and slope.

"""
import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")

import pylops

############################################
# Let's first define the time and space axes as well as some auxiliary input
# parameters that we will use to create a Ricker wavelet
par = {
    "ox": -200,
    "dx": 2,
    "nx": 201,
    "ot": 0,
    "dt": 0.004,
    "nt": 501,
    "f0": 20,
    "nfmax": 210,
}

# Create axis
t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)

# Create wavelet
wav = pylops.utils.wavelets.ricker(np.arange(41) * par["dt"], f0=par["f0"])[0]

############################################
# We want to create a 2d data with a number of crossing linear events using the
# :py:func:`pylops.utils.seismicevents.linear2d` routine.
v = 1500
t0 = [0.2, 0.7, 1.6]
theta = [40, 0, -60]
amp = [1.0, 0.6, -2.0]

mlin, mlinwav = pylops.utils.seismicevents.linear2d(x, t, v, t0, theta, amp, wav)

############################################
# Let's now define the slowness axis and use :py:class:`pylops.signalprocessing.Radon2D`
# to implement our benchmark linear Radon. Refer to the documentation of the
# operator for a more detailed mathematical description of linear Radon.
npx, pxmax = 41, 1e-3
px = np.linspace(-pxmax, pxmax, npx)

RLop = pylops.signalprocessing.Radon2D(
    t, x, px, centeredh=False, kind="linear", interp=False, engine="numpy"
)

# Compute adjoint = Radon transform
mlinwavR = RLop.H * mlinwav.ravel()
mlinwavR = mlinwavR.reshape(npx, par["nt"])

############################################
# Now, let's try to reimplement this operator from scratch using :py:class:`pylops.basicoperators.Spread`.
# We will use the on-the-fly approach, and therefore we are required to create
# function that provided with the indices of the model domain, here :math:`(p_x,t_0)`
# where :math:`p_x` is the slope and :math:`t_0` is the intercept of the
# parametric curve :math:`t(x)=t_0 + p_x x` we wish to spread the model over
# in the data domain.
#
# Note that what we do here can be easily adapted to the case we want to use a
# pre-computed table. We will simply need to run our function upfront for all
# possible :math:`(p_x,t_0)` pairs and store the indices.


def fh(ipx, it0, xaxis, px, nx, ot, dt, nt):
    tx = t[it0] + xaxis * px[ipx]
    it0_frac = (tx - ot) / dt
    itx = np.floor(it0_frac)
    itx[itx > nt - 1] = np.nan
    return itx


xaxis = x
fRad = lambda x, y: fh(x, y, xaxis, px, par["nx"], par["ot"], par["dt"], par["nt"])
RL1op = pylops.Spread((npx, par["nt"]), (par["nx"], par["nt"]), fh=fRad, interp=False)

mlinwavR1 = RL1op.H * mlinwav.ravel()
mlinwavR1 = mlinwavR1.reshape(npx, par["nt"])


fig, axs = plt.subplots(1, 3, figsize=(9, 5))
axs[0].imshow(
    mlinwav.T,
    aspect="auto",
    interpolation="nearest",
    vmin=-1,
    vmax=1,
    cmap="gray",
    extent=(x.min(), x.max(), t.max(), t.min()),
)
axs[0].set_title("Linear events", fontsize=12, fontweight="bold")
axs[0].set_xlabel(r"$x(m)$")
axs[0].set_ylabel(r"$t(s)$")
axs[1].imshow(
    mlinwavR.T,
    aspect="auto",
    interpolation="nearest",
    vmin=-10,
    vmax=10,
    cmap="gray",
    extent=(px.min(), px.max(), t.max(), t.min()),
)
axs[1].set_title("Original Linear Radon", fontsize=12, fontweight="bold")
axs[1].set_xlabel(r"$p_x(s/m)$")
axs[1].set_ylabel(r"$t(s)$")
axs[2].imshow(
    mlinwavR1.T,
    aspect="auto",
    interpolation="nearest",
    vmin=-10,
    vmax=10,
    cmap="gray",
    extent=(px.min(), px.max(), t.max(), t.min()),
)
axs[2].set_title("Re-implemented Linear Radon", fontsize=12, fontweight="bold")
axs[2].set_xlabel(r"$p_x(s/m)$")
axs[2].set_ylabel(r"$t(s)$")
fig.tight_layout()
