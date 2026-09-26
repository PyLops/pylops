r"""
05. Image deblurring
====================
*Deblurring* is the process of removing blurring effects from images, caused for
example by defocus aberration or motion blur.

In forward mode, such blurring effect is typically modelled as a 2-dimensional
convolution between the so-called *point spread function* and a target
sharp input image, where the sharp input image (which has to be recovered) is
unknown and the point-spread function can be either known or unknown.

In this tutorial, an example of 2d blurring and deblurring will be shown using
the :py:class:`pylops.signalprocessing.Convolve2D` operator assuming knowledge
of the point-spread function. Different regularization strategies are compared,
including sparsity-promoting inversion in the Discrete Wavelet Transform (DWT)
and Stationary Wavelet Transform (SWT) domains.
"""

import matplotlib.pyplot as plt
import numpy as np

import pylops
from pylops.utils.metrics import snr

###############################################################################
# Let's start by importing a 2d image and defining the blurring operator
im = np.load("../testdata/python.npy")[::5, ::5, 0].astype(np.float32)

Nz, Nx = im.shape

# Blurring guassian operator
nh = [15, 25]
hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
hz /= np.trapezoid(hz)  # normalize the integral to 1
hx /= np.trapezoid(hx)  # normalize the integral to 1
h = hz[:, np.newaxis] * hx[np.newaxis, :]

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
him = ax.imshow(h)
ax.set_title("Blurring operator")
fig.colorbar(him, ax=ax)
ax.axis("tight")

Cop = pylops.signalprocessing.Convolve2D(
    (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
)

###############################################################################
# We first apply the blurring operator to the sharp image. We then
# try to recover the sharp input image by inverting the convolution operator
# from the blurred image. Note that when we perform inversion without any
# regularization, the deblurred image will show some ringing due to the
# instabilities of the inverse process. Using a L1 solver with a DWT
# preconditioner or TV regularization allows to recover sharper contrasts.
# Finally, replacing the DWT with the SWT, which is shift-invariant and
# redundant, further reduces the blocky artifacts of the Haar wavelet.
imblur = Cop * im

imdeblur = pylops.optimization.leastsquares.normal_equations_inversion(
    Cop,
    imblur.ravel(),
    None,
    maxiter=50,  # solvers need 1D arrays
)[0]
imdeblur = imdeblur.reshape(Cop.dims)

Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)
Dop = [
    pylops.FirstDerivative((Nz, Nx), axis=0, edge=False),
    pylops.FirstDerivative((Nz, Nx), axis=1, edge=False),
]
DWop = Dop + [Wop]

imdeblurfista = pylops.optimization.sparsity.fista(
    Cop * Wop.H, imblur.ravel(), eps=1e-1, niter=100
)[0]
imdeblurfista = imdeblurfista.reshape((Cop * Wop.H).dims)
imdeblurfista = Wop.H * imdeblurfista

Sop = pylops.signalprocessing.SWT2D((Nz, Nx), wavelet="haar", level=3)

imdeblurfistaswt = pylops.optimization.sparsity.fista(
    Cop * Sop.H, imblur.ravel(), eps=3e-2, niter=100
)[0]
imdeblurfistaswt = imdeblurfistaswt.reshape((Cop * Sop.H).dims)
imdeblurfistaswt = Sop.H * imdeblurfistaswt

imdeblurtv = pylops.optimization.sparsity.splitbregman(
    Cop,
    imblur.ravel(),
    Dop,
    niter_outer=10,
    niter_inner=5,
    mu=1.5,
    epsRL1s=[2e0, 2e0],
    tol=1e-4,
    tau=1.0,
    show=False,
    **dict(iter_lim=5, damp=1e-4),
)[0]
imdeblurtv = imdeblurtv.reshape(Cop.dims)

imdeblurtv1 = pylops.optimization.sparsity.splitbregman(
    Cop,
    imblur.ravel(),
    DWop,
    niter_outer=10,
    niter_inner=5,
    mu=1.5,
    epsRL1s=[1e0, 1e0, 1e0],
    tol=1e-4,
    tau=1.0,
    show=False,
    **dict(iter_lim=5, damp=1e-4),
)[0]
imdeblurtv1 = imdeblurtv1.reshape(Cop.dims)

###############################################################################
# Finally we visualize the original, blurred, and recovered images.

# sphinx_gallery_thumbnail_number = 2
fig = plt.figure(figsize=(15, 6))
fig.suptitle("Deblurring", fontsize=14, fontweight="bold", y=0.95)
ax1 = plt.subplot2grid((2, 6), (0, 0))
ax2 = plt.subplot2grid((2, 6), (0, 1))
ax3 = plt.subplot2grid((2, 6), (0, 2))
ax4 = plt.subplot2grid((2, 6), (0, 3))
ax5 = plt.subplot2grid((2, 6), (1, 0))
ax6 = plt.subplot2grid((2, 6), (1, 1))
ax7 = plt.subplot2grid((2, 6), (1, 2))
ax8 = plt.subplot2grid((2, 6), (1, 3))
ax9 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
ax10 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
ax1.axis("tight")
ax1.set_title("Original")
ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
ax2.axis("tight")
ax2.set_title(f"Blurred\nSNR={snr(im, imblur):.2f} dB")
ax3.imshow(imdeblur, cmap="viridis", vmin=0, vmax=250)
ax3.axis("tight")
ax3.set_title(f"L2\nSNR={snr(im, imdeblur):.2f} dB")
ax4.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
ax4.axis("tight")
ax4.set_title(f"FISTA (DWT)\nSNR={snr(im, imdeblurfista):.2f} dB")
ax5.imshow(imdeblurfistaswt, cmap="viridis", vmin=0, vmax=250)
ax5.axis("tight")
ax5.set_title(f"FISTA (SWT)\nSNR={snr(im, imdeblurfistaswt):.2f} dB")
ax6.imshow(imdeblurtv, cmap="viridis", vmin=0, vmax=250)
ax6.axis("tight")
ax6.set_title(f"SB (TV)\nSNR={snr(im, imdeblurtv):.2f} dB")
ax7.imshow(imdeblurtv1, cmap="viridis", vmin=0, vmax=250)
ax7.axis("tight")
ax7.set_title(f"SB (TV+Haar)\nSNR={snr(im, imdeblurtv1):.2f} dB")
ax9.plot(im[Nz // 2], "k")
ax9.plot(imblur[Nz // 2], "--r")
ax9.plot(imdeblur[Nz // 2], "--b")
ax9.plot(imdeblurfista[Nz // 2], "--g")
ax9.plot(imdeblurfistaswt[Nz // 2], "--c")
ax9.plot(imdeblurtv[Nz // 2], "--m")
ax9.plot(imdeblurtv1[Nz // 2], "--y")
ax9.axis("tight")
ax9.set_title("Horizontal section")
ax10.plot(im[:, Nx // 2], "k", label="Original")
ax10.plot(imblur[:, Nx // 2], "--r", label="Blurred")
ax10.plot(imdeblur[:, Nx // 2], "--b", label="L2")
ax10.plot(imdeblurfista[:, Nx // 2], "--g", label="FISTA (DWT)")
ax10.plot(imdeblurfistaswt[:, Nx // 2], "--c", label="FISTA (SWT)")
ax10.plot(imdeblurtv[:, Nx // 2], "--m", label="SB (TV)")
ax10.plot(imdeblurtv1[:, Nx // 2], "--y", label="SB (TV+Haar)")
ax10.axis("tight")
ax10.set_title("Vertical section")
ax8.legend(*ax10.get_legend_handles_labels(), loc="center", fontsize="small")
ax8.axis("off")
plt.tight_layout()
plt.subplots_adjust(top=0.8)
