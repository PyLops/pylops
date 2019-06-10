"""
Roll
====
This example shows how to use the :py:class:`pylops.Roll` operator.

This operator simply shifts elements of multi-dimensional array along a
specified direction a chosen number of samples.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start with a 1d example. We make a signal, shift it by two samples
# and then shift it back using its adjoint. We can immediately see how the
# adjoint of this operator is equivalent to its inverse.
nx = 10
x = np.arange(nx)

Rop = pylops.Roll(nx, shift=2)

y = Rop*x
xadj = Rop.H*y

plt.figure()
plt.plot(x, 'k', lw=2,  label='x')
plt.plot(y, 'b', lw=2, label='y')
plt.plot(xadj, '--r', lw=2, label='xadj')
plt.title('1D Roll')
plt.legend()

###############################################################################
# We can now do the same with a 2d array.
ny, nx = 10, 5
x = np.arange(ny*nx).reshape(ny, nx)

Rop = pylops.Roll(ny*nx, dims=(ny, nx), dir=1, shift=-2)

y = Rop*x.ravel()
xadj = Rop.H*y

y = y.reshape(ny, nx)
xadj = xadj.reshape(ny, nx)

fig, axs = plt.subplots(1, 3, figsize=(10, 2))
fig.suptitle('Roll for 2d data', fontsize=14, fontweight='bold', y=1.15)
axs[0].imshow(x, cmap='rainbow', vmin=0, vmax=50)
axs[0].set_title(r'$x$')
axs[0].axis('tight')
axs[1].imshow(y, cmap='rainbow', vmin=0, vmax=50)
axs[1].set_title(r'$y = R x$')
axs[1].axis('tight')
axs[2].imshow(xadj, cmap='rainbow', vmin=0, vmax=50)
axs[2].set_title(r'$x_{adj} = R^H y$')
axs[2].axis('tight')
