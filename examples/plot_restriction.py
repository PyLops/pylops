"""
Restriction
===========

This example shows how to use the :py:class:`pylops.Restriction` operator
to perform *Sampling* of a certain input vector at locations ``iava``.


As explained in the **Optimization** tutorial, such operators can be used to interpolate irregularly
sampled 1d or 2d signals onto a regular grid.
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
# We then create the restriction operator and display the original signal as well
# as the subsampled signal.

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
