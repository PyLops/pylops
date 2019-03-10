r"""
2D Sliding
==========
This example shows how to use the :py:class:`pylops.signalprocessing.Sliding2D`
operator to perform repeated transforms over small patches of a 2-dimensional
array. The transform that we apply in this example is the
:py:class:`pylops.signalprocessing.Radon2D` but this operator has been
design to allow a variety of transforms as long as they operate with signals
that are 2-dimensional in nature.

"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by creating an 2-dimensional array of size :math:`n_x \times n_t`
# composed of 3 parabolic events
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
# We want to divide this 2-dimensional data into small overlapping
# patches in the spatial direction and apply the adjoint of the
# :py:class:`pylops.signalprocessing.Radon2D` operator to each patch. This is
# done by simply using the adjoint of the
# :py:class:`pylops.signalprocessing.Sliding2D` operator
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
# We now create  a similar operator but we also add a taper to the overlapping
# parts of the patches.
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
# :py:class:`pylops.signalprocessing.Sliding2D` operator.
fig, axs = plt.subplots(2, 3, sharey=True, figsize=(12, 10))
im = axs[0][0].imshow(data.T, cmap='gray')
axs[0][0].set_title('Original data')
plt.colorbar(im, ax=axs[0][0])
axs[0][0].axis('tight')
im = axs[0][1].imshow(radon.T, cmap='gray')
axs[0][1].set_title('Adjoint Radon')
plt.colorbar(im, ax=axs[0][1])
axs[0][1].axis('tight')
im = axs[0][2].imshow(reconstructed_data.T, cmap='gray')
axs[0][2].set_title('Reconstruction from adjoint')
plt.colorbar(im, ax=axs[0][2])
axs[0][2].axis('tight')
axs[1][0].axis('off')
im = axs[1][1].imshow(radoninv.T, cmap='gray')
axs[1][1].set_title('Inverse Radon')
plt.colorbar(im, ax=axs[1][1])
axs[1][1].axis('tight')
im = axs[1][2].imshow(reconstructed_datainv.T, cmap='gray')
axs[1][2].set_title('Reconstruction from inverse')
plt.colorbar(im, ax=axs[1][2])
axs[1][2].axis('tight')

for i in range(0, 114, 24):
    axs[0][0].axvline(i, color='w', lw=1, ls='--')
    axs[0][0].axvline(i + winsize, color='k', lw=1, ls='--')
    axs[0][0].text(i + winsize//2, par['nt']-10, 'w'+str(i//24),
                   ha='center', va='center', weight='bold',
                   color='w')

for i in range(0, 305, 61):
    axs[0][1].axvline(i, color='w', lw=1, ls='--')
    axs[0][1].text(i + npx//2, par['nt']-10, 'w'+str(i//61),
                   ha='center', va='center', weight='bold',
                   color='w')
    axs[1][1].axvline(i, color='w', lw=1, ls='--')
    axs[1][1].text(i + npx//2, par['nt']-10, 'w'+str(i//61),
                   ha='center', va='center', weight='bold',
                   color='w')

###############################################################################
# We notice two things, i)provided small enough patches and a transform
# that can explain data *locally*, we have been able reconstruct our
# original data almost to perfection. ii) inverse is betten than adjoint as
# expected as the adjoin does not only introduce small artifacts but also does
# not respect the original amplitudes of the data.
#
# An appropriate transform alongside with a sliding window approach will
# result a very good approach for interpolation (or *regularization*) or
# irregularly sampled seismic data.
