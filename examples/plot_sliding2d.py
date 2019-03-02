r"""
2D Sliding
==========
This example shows how to use the :py:class:`pylops.signalprocessing.Sliding2D`
operator to perform repeated transforms over small patches of a two dimensional
signal. The transform that we apply in this example is the
:py:class:`pylops.signalprocessing.Radon2D` but this operator has been
design to allow a variety of transforms as long as they operate with signals
that are two dimensional in nature.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by creating an 2d matrix of size :math:`n_x \times n_t`
# and composed of 3 parabolic events
par = {'ox':-140, 'dx':2, 'nx':140,
       'ot':0, 'dt':0.004, 'nt':200,
       'f0': 20}

v = 1500
t0 = [0.2, 0.4, 0.5]
px = [0, 0, 0]
pxx = [1e-5, 5e-6, 1e-20]
amp = [1., -2, 0.5]

# Create axis
t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)

# Create wavelet
wav = pylops.utils.wavelets.ricker(t[:41], f0=par['f0'])[0]

# Generate model
_, data = pylops.utils.seismicevents.parabolic2d(x, t, t0, px,
                                                 pxx, amp, wav)

###############################################################################
# We start by divide this 2d dimensional data into small overlapping
# patches in the spatial direction and apply the adjoint
# :py:class:`pylops.signalprocessing.Radon2D` to each patch. This is done by
# using the adjoint of the :py:class:`pylops.signalprocessing.Sliding2D`
# operator
nwins = 5
winsize = 36
overlap = 10
npx = 61
px = np.linspace(-5e-3, 5e-3, npx)

dimsd = data.shape
dims = (nwins*npx, par['nt'])

# sliding window transform without taper
Op = \
    pylops.signalprocessing.Radon2D(t, np.linspace(-par['dx']*winsize//2,
                                                   par['dx']*winsize//2,
                                                   winsize),
                                    px, centeredh=True, kind='linear',
                                    engine='numba')
Slid = pylops.signalprocessing.Sliding2D(Op, dims, dimsd,
                                         winsize, overlap,
                                         tapertype=None)

radon = Slid.H * data.flatten()
radon = radon.reshape(dims)

###############################################################################
# We want to show now how we can simply apply the forward of the same operator
# (this time adding a taper in the overllapping part of the patches)
Slid = pylops.signalprocessing.Sliding2D(Op, dims, dimsd,
                                         winsize, overlap,
                                         tapertype='cosine')

reconstructed_data = Slid * radon.flatten()
reconstructed_data = reconstructed_data.reshape(dimsd)

###############################################################################
# We will see that our reconstructed signal presents some small artifacts.
# This is because we have not inverted our operator but simply applied
# the adjoint to estimate the representation of the input data in the Radon
# domain. We can do better if we use the inverse instead.
radoninv = pylops.LinearOperator(Slid, explicit=False).div(data.flatten(),
                                                           niter=10)
reconstructed_datainv = Slid * radoninv.flatten()

radoninv = radoninv.reshape(dims)
reconstructed_datainv = reconstructed_datainv.reshape(dimsd)

###############################################################################
# Let's finally visualize all the intermediate results as well as our final
# data reconstruction after inverting the
# :py:class:`pylops.signalprocessing.Sliding2D` operator. As you can see,
# provided small enough patches and a transform that can explain data
# *locally*, we have been able reconstruct our original data almost to
# perfection. An appropriate transform and a sliding window approach will
# result a very good approach for interpolation (or *regularization*) or
# irregularly sampled seismic data.
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
axs[0][0].imshow(data.T, cmap='gray')
axs[0][0].set_title('Original data')
axs[0][0].axis('tight')
axs[0][1].imshow(radon.T, cmap='gray')
axs[0][1].set_title('Adjoint Radon')
axs[0][1].axis('tight')
axs[0][2].imshow(reconstructed_data.T, cmap='gray')
axs[0][2].set_title('Reconstruction from adjoint')
axs[0][2].axis('tight')
axs[1][0].axis('off')
axs[1][1].imshow(radoninv.T, cmap='gray')
axs[1][1].set_title('Inverse Radon')
axs[1][1].axis('tight')
axs[1][2].imshow(reconstructed_datainv.T, cmap='gray')
axs[1][2].set_title('Reconstruction from inverse')
axs[1][2].axis('tight')
