r"""
18. Deblending
==============
The cocktail party problem arises when sounds from different sources mix before reaching our ears
(or any recording device), requiring the brain (or any hardware in the recording device) to estimate
individual sources from the received mixture. In seismic acquisition, an analog problem is present
when multiple sources are fired simultaneously. This family of acquisition methods is usually referred to as
simultaneous shooting and the problem of separating the blendend shot gathers into their individual
components is called deblending. Whilst various firing strategies can be adopted, in this example
we consider the continuos blending problem where a single source is fired sequentially at an interval
shorter than the amount of time required for waves to travel into the Earth and come back.

Simply stated the forward problem can be written as:

.. math::
    \mathbf{d}^b = \boldsymbol\Phi \mathbf{d}

Here :math:`\mathbf{d} = [\mathbf{d}_1^T, \mathbf{d}_2^T,\ldots,
\mathbf{d}_N^T]^T` is a stack of :math:`N` individual shot gathers,
:math:`\boldsymbol\Phi=[\boldsymbol\Phi_1, \boldsymbol\Phi_2,\ldots,
\boldsymbol\Phi_N]` is the blending operator, :math:`\mathbf{d}^b` is the
so-called supergather than contains all shots superimposed to each other.

In order to successfully invert this severely underdetermined problem, two key
ingredients must be introduced:

- the firing time of each source (i.e., shifts of the blending operator) must be
  chosen to be dithered around a nominal regular, periodic firing interval.
  In our case, we consider shots of duration :math:`T=4s`, regular firing time of :math:`T_s=2s`
  and a dithering code as follows :math:`\Delta t = U(-1,1)`;
- prior information about the data to reconstruct, either in the form of regularization
  or preconditioning must be introduced. In our case we will use a patch-FK transform
  as preconditioner and solve the problem imposing sparsity in the transformed domain.

In other words, we aim to solve the following problem:

.. math::
    J = \|\mathbf{d}^b - \boldsymbol\Phi \mathbf{S}^H \mathbf{x}\|_2 + \epsilon \|\mathbf{x}\|_1

for which we will use the :py:class:`pylops.optimization.sparsity.FISTA` solver.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lobpcg as sp_lobpcg

import pylops

np.random.seed(10)
plt.close("all")


###############################################################################
# Let's start by defining a blending operator
def Blending(nt, ns, dt, overlap, times, dtype="float64"):
    """Blending operator"""
    pad = int(overlap * nt)
    OpShiftPad = []
    for i in range(ns):
        PadOp = pylops.Pad(nt, (pad * i, pad * (ns - 1 - i)), dtype=dtype)
        ShiftOp = pylops.signalprocessing.Shift(
            pad * (ns - 1) + nt, times[i], axis=0, sampling=dt, real=False, dtype=dtype
        )
        OpShiftPad.append(ShiftOp * PadOp)
    return pylops.HStack(OpShiftPad)


###############################################################################
# We can now load and display a small portion of the MobilAVO dataset composed
# of 60 shots and a single receiver. This data is unblended.

data = np.load("../testdata/deblending/mobil.npy")
ns, nt = data.shape

dt = 0.004
t = np.arange(nt) * dt

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(
    data.T,
    cmap="gray",
    vmin=-50,
    vmax=50,
    extent=(0, ns, t[-1], 0),
    interpolation="none",
)
ax.set_title("CRG")
ax.set_xlabel("#Src")
ax.set_ylabel("t [s]")
ax.axis("tight")
plt.tight_layout()

###############################################################################
# We are now ready to define the blending operator, blend our data, and apply
# the adjoint of the blending operator to it. This is usually referred as
# pseudo-deblending: as we will see brings back each source to its own nominal
# firing time, but since sources partially overlap in time, it will also generate
# some burst like noise in the data. Deblending can hopefully fix this.

overlap = 0.5
pad = int(overlap * nt)
ignition_times = 2.0 * np.random.rand(ns) - 1.0
Bop = Blending(nt, ns, dt, overlap, ignition_times, dtype="complex128")
data_blended = Bop * data.ravel()
data_pseudo = Bop.H * data_blended.ravel()
data_pseudo = data_pseudo.reshape(ns, nt)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(
    data_pseudo.T.real,
    cmap="gray",
    vmin=-50,
    vmax=50,
    extent=(0, ns, t[-1], 0),
    interpolation="none",
)
ax.set_title("Pseudo-deblended CRG")
ax.set_xlabel("#Src")
ax.set_ylabel("t [s]")
ax.axis("tight")
plt.tight_layout()

###############################################################################
# We are finally ready to solve our deblending inverse problem

# Patched FK
dimsd = data.shape
nwin = (20, 80)
nover = (10, 40)
nop = (128, 128)
nop1 = (128, 65)
nwins = (5, 24)
dims = (nwins[0] * nop1[0], nwins[1] * nop1[1])

Fop = pylops.signalprocessing.FFT2D(nwin, nffts=nop, real=True)
Sop = pylops.signalprocessing.Patch2D(
    Fop.H, dims, dimsd, nwin, nover, nop1, tapertype="hanning", design=False
)
# Overall operator
Op = Bop * Sop

# Compute max eigenvalue (we do this explicitly to be able to run this fast)
Op1 = pylops.LinearOperator(Op.H * Op, explicit=False)
X = np.random.rand(Op1.shape[0], 1).astype(Op1.dtype)
maxeig = sp_lobpcg(Op1, X=X, maxiter=5, tol=1e-10)[0][0]
alpha = 1.0 / maxeig

# Deblend
niter = 60
decay = (np.exp(-0.05 * np.arange(niter)) + 0.2) / 1.2

with pylops.disabled_ndarray_multiplication():
    p_inv = pylops.FISTA(
        Op,
        data_blended.ravel(),
        niter=niter,
        eps=5e0,
        alpha=alpha,
        decay=decay,
        show=True,
        returninfo=True,
    )[0]
data_inv = Sop * p_inv
data_inv = data_inv.reshape(ns, nt)

fig, axs = plt.subplots(1, 4, sharey=False, figsize=(12, 8))
axs[0].imshow(
    data.T.real,
    cmap="gray",
    extent=(0, ns, t[-1], 0),
    vmin=-50,
    vmax=50,
    interpolation="none",
)
axs[0].set_title("CRG")
axs[0].set_xlabel("#Src")
axs[0].set_ylabel("t [s]")
axs[0].axis("tight")
axs[1].imshow(
    data_pseudo.T.real,
    cmap="gray",
    extent=(0, ns, t[-1], 0),
    vmin=-50,
    vmax=50,
    interpolation="none",
)
axs[1].set_title("Pseudo-deblended CRG")
axs[1].set_xlabel("#Src")
axs[1].axis("tight")
axs[2].imshow(
    data_inv.T.real,
    cmap="gray",
    extent=(0, ns, t[-1], 0),
    vmin=-50,
    vmax=50,
    interpolation="none",
)
axs[2].set_xlabel("#Src")
axs[2].set_title("Deblended CRG")
axs[2].axis("tight")
axs[3].imshow(
    data.T.real - data_inv.T.real,
    cmap="gray",
    extent=(0, ns, t[-1], 0),
    vmin=-50,
    vmax=50,
    interpolation="none",
)
axs[3].set_xlabel("#Src")
axs[3].set_title("Blending error")
axs[3].axis("tight")
plt.tight_layout()

###############################################################################
# Finally, let's look a bit more at what really happened under the hood. We
# display a number of patches and their associated FK spectrum

Sop1 = pylops.signalprocessing.Patch2D(
    Fop.H, dims, dimsd, nwin, nover, nop1, tapertype=None, design=False
)

# Original
p = Sop1.H * data.ravel()
preshape = p.reshape(nwins[0], nwins[1], nop1[0], nop1[1])

ix = 16
fig, axs = plt.subplots(2, 4, figsize=(12, 5))
fig.suptitle("Data patches")
for i in range(4):
    axs[0][i].imshow(np.fft.fftshift(np.abs(preshape[i, ix]).T, axes=1))
    axs[0][i].axis("tight")
    axs[1][i].imshow(
        np.real((Fop.H * preshape[i, ix].ravel()).reshape(nwin)).T,
        cmap="gray",
        vmin=-30,
        vmax=30,
        interpolation="none",
    )
    axs[1][i].axis("tight")
plt.tight_layout()

# Pseudo-deblended
p_pseudo = Sop1.H * data_pseudo.ravel()
p_pseudoreshape = p_pseudo.reshape(nwins[0], nwins[1], nop1[0], nop1[1])

ix = 16
fig, axs = plt.subplots(2, 4, figsize=(12, 5))
fig.suptitle("Pseudo-deblended patches")
for i in range(4):
    axs[0][i].imshow(np.fft.fftshift(np.abs(p_pseudoreshape[i, ix]).T, axes=1))
    axs[0][i].axis("tight")
    axs[1][i].imshow(
        np.real((Fop.H * p_pseudoreshape[i, ix].ravel()).reshape(nwin)).T,
        cmap="gray",
        vmin=-30,
        vmax=30,
        interpolation="none",
    )
    axs[1][i].axis("tight")
plt.tight_layout()

# Deblended
p_inv = Sop1.H * data_inv.ravel()
p_invreshape = p_inv.reshape(nwins[0], nwins[1], nop1[0], nop1[1])

ix = 16
fig, axs = plt.subplots(2, 4, figsize=(12, 5))
fig.suptitle("Deblended patches")
for i in range(4):
    axs[0][i].imshow(np.fft.fftshift(np.abs(p_invreshape[i, ix]).T, axes=1))
    axs[0][i].axis("tight")
    axs[1][i].imshow(
        np.real((Fop.H * p_invreshape[i, ix].ravel()).reshape(nwin)).T,
        cmap="gray",
        vmin=-30,
        vmax=30,
        interpolation="none",
    )
    axs[1][i].axis("tight")
plt.tight_layout()
