r"""
4. Image deblurring
===================
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
fig = plt.figure(figsize=(8, 5))
fig.suptitle('Deblurring', fontsize=14, fontweight='bold', y=0.95)
ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=3)
ax2 = plt.subplot2grid((5, 4), (0, 1), rowspan=3)
ax3 = plt.subplot2grid((5, 4), (0, 2), rowspan=3)
ax4 = plt.subplot2grid((5, 4), (0, 3), rowspan=3)
ax5 = plt.subplot2grid((5, 4), (3, 0), rowspan=2, colspan=2)
ax6 = plt.subplot2grid((5, 4), (3, 2), rowspan=2, colspan=2)
ax1.imshow(im, cmap='viridis', vmin=0, vmax=250)
ax1.axis('tight')
ax1.set_title('Original')
ax2.imshow(imblur, cmap='viridis', vmin=0, vmax=250)
ax2.axis('tight')
ax2.set_title('Blurred')
ax3.imshow(imdeblur, cmap='viridis', vmin=0, vmax=250)
ax3.axis('tight')
ax3.set_title('Deblurred')
ax4.imshow(imdeblurreg, cmap='viridis', vmin=0, vmax=250)
ax4.axis('tight')
ax4.set_title('Regularized deblurred')
ax5.plot(im[Nz//2], 'k')
ax5.plot(imblur[Nz//2], '--r')
ax5.plot(imdeblur[Nz//2], '--b')
ax5.plot(imdeblurreg[Nz//2], '--g')
ax5.set_title('Horizontal section')
ax6.plot(im[:, Nx//2], 'k', label='Original')
ax6.plot(imblur[:, Nx//2], '--r', label='Blurred')
ax6.plot(imdeblur[:, Nx//2], '--b', label='Deblurred')
ax6.plot(imdeblurreg[:, Nx//2], '--g', label='Regularized deblurred')
ax6.set_title('Vertical section')
ax6.legend(loc=5, fontsize='small')
plt.tight_layout()
plt.subplots_adjust(top=0.8)
