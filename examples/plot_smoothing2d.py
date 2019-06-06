"""
2D Smoothing
============

This example shows how to use the :py:class:`pylops.Smoothing2D` operator
to smooth a multi-dimensional input signal along two given axes.

"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Define the input parameters: number of samples of input signal (``N`` and ``M``) and
# lenght of the smoothing filter regression coefficients
# (:math:`n_{smooth,1}` and :math:`n_{smooth,2}`). In this first case the input
# signal is one at the center and zero elsewhere.
N, M = 11, 21
nsmooth1, nsmooth2 = 5, 3
A = np.zeros((N, M))
A[5, 10] = 1

Sop = pylops.Smoothing2D(nsmooth=[nsmooth1, nsmooth2], dims=[N, M], dtype='float64')
B = Sop*A.flatten()
B = np.reshape(B, (N, M))

###############################################################################
# After applying smoothing, we will also try to invert it.
Aest = Sop/B.flatten()
Aest = np.reshape(Aest, (N, M))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
im = axs[0].imshow(A, interpolation='nearest', vmin=0, vmax=1)
axs[0].axis('tight')
axs[0].set_title('Model')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B, interpolation='nearest', vmin=0, vmax=1)
axs[1].axis('tight')
axs[1].set_title('Data')
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(Aest, interpolation='nearest', vmin=0, vmax=1)
axs[2].axis('tight')
axs[2].set_title('Estimated model')
plt.colorbar(im, ax=axs[2])
