"""
Restriction
===========

This example shows how to use the :py:class:`pylops.Restriction` operator
to sample a certain input vector at desired locations ``iava``.

As explained in the :ref:`sphx_glr_tutorials_solvers.py` tutorial, such
operators can be used as forward model in an inverse problem aimed at
interpolate irregularly sampled 1d or 2d signals onto a regular grid.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's create a signal of size ``nt`` and sampling ``dt`` that is composed
# of three sinusoids at frequencies ``freqs``.
nt = 200
dt = 0.004

freqs = [5., 3., 8.]

t = np.arange(nt)*dt
x = np.zeros(nt)

for freq in freqs:
    x = x + np.sin(2*np.pi*freq*t)

###############################################################################
# First of all, we subsample the signal at random locations and we retain 40%
# of the initial samples.
perc_subsampling = 0.4
ntsub = int(np.round(nt*perc_subsampling))

iava = np.sort(np.random.permutation(np.arange(nt))[:ntsub])

###############################################################################
# We then create the restriction operator and display the original signal as
# well as the subsampled signal.

Rop = pylops.Restriction(nt, iava, dtype='float64')

y = Rop*x
ymask = Rop.mask(x)

# Visualize data
fig = plt.figure(figsize=(15, 5))
plt.plot(t, x, 'k', lw=3)
plt.plot(t, x, '.k', ms=20, label='all samples')
plt.plot(t, ymask, '.g', ms=15, label='available samples')
plt.legend()
plt.title('Data restriction')

###############################################################################
# Finally we show how the :py:class:`pylops.Restriction` is not limited to
# one dimensional signals but can be applied to sample locations of a specific
# axis of a multi-dimensional array.
# subsampling locations
nx, nt = 100, 50

x = np.random.normal(0, 1, (nx, nt))

perc_subsampling = 0.4
nxsub = int(np.round(nx*perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(nx))[:nxsub])

Rop = pylops.Restriction(nx*nt, iava, dims=(nx, nt), dir=0, dtype='float64')
y = (Rop*x.ravel()).reshape(nxsub, nt)
ymask = Rop.mask(x)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(x.T, cmap='gray')
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(y.T, cmap='gray')
axs[1].set_title('Data')
axs[1].axis('tight')
axs[2].imshow(ymask.T, cmap='gray')
axs[2].set_title('Masked model')
axs[2].axis('tight')
