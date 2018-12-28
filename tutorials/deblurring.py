r"""
Image deblurring
================
*Deblurring* is the process of removing blurring effects from images, caused for
example by defocus aberration or motion blur.

In forward mode, such blurring effect is typically modelled as a 2-dimensional
convolution between the so-called *point spread function* and a target
sharp input image, where the sharp input image (which has to be recovered) is
unknown and the point-spread function can be either known or unknown.

In this tutorial, an example of 2d blurring and deblurring will be shown using
the :py:class:`pylops.signalprocessing.Convolve2D` operator assuming knownledge
of the point-spread function.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import pylops

###############################################################################
# Let's start by importing a 2d image and defining the blurring operator
im = np.asarray(misc.imread('../testdata/python.png'))[::5, ::5, 0]

Nz, Nx = im.shape

# Blurring guassian operator
nh = [15, 25]
hz = np.exp(-0.1*np.linspace(-(nh[0]//2), nh[0]//2, nh[0])**2)
hx = np.exp(-0.03*np.linspace(-(nh[1]//2), nh[1]//2, nh[1])**2)
hz /= np.trapz(hz) # normalize the integral to 1
hx /= np.trapz(hx) # normalize the integral to 1
h = hz[:, np.newaxis] * hx[np.newaxis, :]

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
him = ax.imshow(h)
ax.set_title('Blurring operator')
fig.colorbar(him, ax=ax)
ax.axis('tight')

Cop = pylops.signalprocessing.Convolve2D(Nz * Nx, h=h,
                                         offset=(nh[0] // 2,
                                                 nh[1] // 2),
                                         dims=(Nz, Nx), dtype='float32')

###############################################################################
# We will now apply the blurring operator to the sharp image. Finally we
# try to recover the sharp input image by inverting the convolution operator
# from the blurred image. Note that when we perform inversion without any
# regularization, the deblurred image will show some ringing due to the
# instabilities of the inverse process. Adding a small Tikhonov damping
# improves the quality of the deblurred image.
imblur = Cop * im.flatten()
imdeblur = \
    pylops.optimization.leastsquares.NormalEquationsInversion(Cop, None,
                                                              imblur,
                                                              maxiter=50)
imdeblurreg = \
    pylops.optimization.leastsquares.NormalEquationsInversion(Cop, None,
                                                              imblur, epsI=0.1,
                                                              maxiter=50)

# Reshape images
imblur = imblur.reshape((Nz, Nx))
imdeblur = imdeblur.reshape((Nz, Nx))
imdeblurreg = imdeblurreg.reshape((Nz, Nx))

###############################################################################
# Finally we visualize the original, blurred, and recovered images.

# sphinx_gallery_thumbnail_number = 2
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
fig.suptitle('Deblurring', fontsize=14,
             fontweight='bold', y=0.95)
axs[0].imshow(im, cmap='viridis', vmin=0, vmax=250)
axs[0].axis('tight')
axs[0].set_title('Original')
axs[1].imshow(imblur, cmap='viridis', vmin=0, vmax=250)
axs[1].axis('tight')
axs[1].set_title('Blurred')
axs[2].imshow(imdeblur, cmap='viridis', vmin=0, vmax=250)
axs[2].axis('tight')
axs[2].set_title('Deblurred')
axs[3].imshow(imdeblurreg, cmap='viridis', vmin=0, vmax=250)
axs[3].axis('tight')
axs[3].set_title('Regularized deblurred')
plt.tight_layout()
plt.subplots_adjust(top=0.8)

