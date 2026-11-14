r"""
PWD-based slope estimation and structural smoothing
===================================================

This example shows how to estimate local slopes of a two-dimensional
array using the Plane-Wave Destruction (PWD [1]_) algorithm via
:py:func:`pylops.utils.signalprocessing.pwd_slope_estimate`; such slopes are
then used as a guide to smooth a noise realization following the structural
dips of the input data via the :py:class:`pylops.signalprocessing.PWSmoother2D`.

.. [1] Fomel, S., "Applications of plane-wave destruction filters",
   Geophysics. 2002.

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylops.signalprocessing import PWSmoother2D
from pylops.utils.signalprocessing import pwd_slope_estimate, slope_estimate

plt.close("all")
np.random.seed(10)


def create_colorbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    return cax, cb


###############################################################################
# Sigmoid model
# ------------------
# To start we import the same `2D image <http://ahay.org/blog/2014/10/08/program-of-the-month-sfsigmoid/>`_
# that we used in the seislet transform example. This image contains curved
# reflectors that we will try to follow during the smoothing operation.
inputfile = "../testdata/sigmoid.npz"

sigmoid = 1e3 * np.load(inputfile)["sigmoid"].T
nz, nx = sigmoid.shape

###############################################################################
# Slope estimation comparison between PWD and Structure Tensor
# ------------------------------------------------------------
# Next, slopes are estimated using both the plane-wave destruction
# and the structure-tensor algorithms. Both algorithms return slopes
# in samples per trace.
pwd_slope = pwd_slope_estimate(
    sigmoid,
    niter=5,
    liter=20,
    order=2,
    smoothing="triangle",
    nsmooth=(12, 12),
    damp=6e-4,
).astype(np.float32)

st_slope, _ = slope_estimate(
    sigmoid,
    dz=1.0,
    dx=1.0,
    smooth=2.0,
    eps=2e-3,
    dips=False,
)
# Defined with z-point upwards, reverse the convention
st_slope *= -1
st_slope = st_slope.astype(np.float32)

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(sigmoid, aspect="auto", cmap="gray")
create_colorbar(im0, ax=ax[0])
ax[0].set_title("Sigmoid model")

v = np.max(np.abs(pwd_slope))
im1 = ax[1].imshow(pwd_slope, aspect="auto", cmap="turbo", vmin=-v, vmax=v)
ax[1].set_title("PWD slope estimate")
create_colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(st_slope, aspect="auto", cmap="turbo", vmin=-v, vmax=v)
ax[2].set_title("Structure-tensor slope estimate")
create_colorbar(im2, ax=ax[2])

for a in ax:
    a.set_xlabel("x (samples)")
ax[0].set_ylabel("z (samples)")
fig.tight_layout()

###############################################################################
# Structure-aligned smoothing via local slopes
# --------------------------------------------
# The estimated slopes are finally used by the
# :py:class:`pylops.signalprocessing.PWSmoother2D` operator to perform
# structure-aligned smoothing of a random noise realization. Note that
# this operator is defined as the composition of a
# :py:class:`pylops.signalprocessing.PWSprayer2D` operator and its adjoint
# (i.e., ``Sprayer.T @ Sprayer``) and therefore it is
# a symmetric operator.
noise = np.random.uniform(-1.0, 1.0, size=(nz, nx)).astype(np.float32)

radius = 6
alpha = 0.7

SOp_pwd = PWSmoother2D(
    dims=(nz, nx), sigma=pwd_slope, radius=radius, alpha=alpha, dtype="float32"
)
smooth_pwd = SOp_pwd @ noise

SOp_st = PWSmoother2D(
    dims=(nz, nx), sigma=st_slope, radius=radius, alpha=alpha, dtype="float32"
)
smooth_st = SOp_st @ noise

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(noise, aspect="auto", cmap="magma")
ax[0].set_title("Random field")
create_colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(smooth_pwd, aspect="auto", cmap="magma")
ax[1].set_title("Smoothed with PWD slopes")
create_colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(smooth_st, aspect="auto", cmap="magma")
ax[2].set_title("Smoothed with structure-tensor slopes")
create_colorbar(im2, ax=ax[2])

for a in ax:
    a.set_xlabel("x (samples)")
ax[0].set_ylabel("z (samples)")
fig.tight_layout()
