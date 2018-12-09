"""
Derivatives
===========
This example shows how to use the suite of derivative operators, namely
:py:class:`pylops.FirstDerivative`, :py:class:`pylops.SecondDerivative`
and :py:class:`pylops.Laplacian`.

The derivative operators are very useful when the model to be inverted for is expect to be
smooth in one or more directions. As shown in the *Optimization* tutorial, these operators
will be used as part of the regularization term to obtain a smooth solution.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by looking at a simple first-order centered derivative and how could implement it
# naively by creating a dense matrix. Note that we will not apply the derivative where the
# stencil is partially outside of the range of the input signal (i.e., at the edge of the signal)
nx = 10

D = np.diag(0.5*np.ones(nx-1), k=1) - np.diag(0.5*np.ones(nx-1), -1)
D[0] = D[-1] = 0

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
im = plt.imshow(D, cmap='rainbow', vmin=-0.5, vmax=0.5)
ax.set_title('First derivative', size=20, fontweight='bold')
ax.set_xticks(np.arange(nx-1)+0.5)
ax.set_yticks(np.arange(nx-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
fig.colorbar(im, ax=ax, ticks=[-0.5, 0.5], shrink=0.7)

###############################################################################
# We now create a signal filled with zero and a single one at its center and apply
# the derivative matrix by means of a dot product
x = np.zeros(nx)
x[int(nx/2)] = 1

y_dir = np.dot(D, x)
xadj_dir = np.dot(D.T, y_dir)

###############################################################################
# Let's now do the same using the :py:class:`pylops.FirstDerivative` operator and compare its
# outputs after applying the forward and adjoint operators to those from the dense matrix.

D1op = pylops.FirstDerivative(nx, dtype='float32')

y_lop = D1op*x
xadj_lop = D1op.H*y_lop

fig, axs = plt.subplots(2, 1, figsize=(13, 5))
axs[0].stem(np.arange(nx), y_dir, 'k', label='direct')
axs[0].stem(np.arange(nx), y_lop, '--r', label='lop')
axs[0].set_title('Forward', size=20, fontweight='bold')
axs[1].stem(np.arange(nx), xadj_dir, 'k', label='direct')
axs[1].stem(np.arange(nx), xadj_lop, '--r', label='lop')
axs[1].set_title('Adjoint', size=20, fontweight='bold')

#############################################
# As expected we obtain the same result, with the only difference that in the second
# case we did not need to explicitly create a matrix, saving memory and computational time.
#
# Let's move onto applying the same first derivative to a 2d array in the first direction
nx, ny = 11, 21
A = np.zeros((nx, ny))
A[nx//2, ny//2] = 1.

D1op = pylops.FirstDerivative(nx * ny, dims=(nx, ny), dir=0, dtype='float64')
B = np.reshape(D1op*np.ndarray.flatten(A), (nx, ny))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('First Derivative in 1st direction', fontsize=12, fontweight='bold')
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y')
plt.colorbar(im, ax=axs[1])
fig.tight_layout()

#############################################
# We can now do the same for the second derivative

A = np.zeros((nx, ny ))
A[nx//2, ny//2] = 1.

D2op = pylops.SecondDerivative(nx * ny, dims=(nx, ny), dir=0, dtype='float64')
B = np.reshape(D2op*np.ndarray.flatten(A), (nx, ny))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Second Derivative in 1st direction', fontsize=12, fontweight='bold')
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y')
plt.colorbar(im, ax=axs[1])

#############################################
# We can also apply the second derivative to the second direction of our data (``dir=1``)
D2op = pylops.SecondDerivative(nx * ny, dims=(nx, ny), dir=1, dtype='float64')
B = np.reshape(D2op*np.ndarray.flatten(A), (nx, ny))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Second Derivative in 2nd direction', fontsize=12, fontweight='bold')
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y')
plt.colorbar(im, ax=axs[1])


#############################################
# And finally we use the symmetrical Laplacian operator as well as a asymmetrical
# version of it (by adding more weight to the derivative along one direction)

# symmetrical
L2symop = pylops.Laplacian(dims=(nx, ny), weights=(1, 1), dtype='float64')

# asymmetrical
L2asymop = pylops.Laplacian(dims=(nx, ny), weights=(3, 1), dtype='float64')

Bsym = np.reshape(L2symop*np.ndarray.flatten(A), (nx, ny))
Basym = np.reshape(L2asymop*np.ndarray.flatten(A), (nx, ny))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
fig.suptitle('Laplacian', fontsize=12, fontweight='bold')
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Bsym, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y sym')
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(Basym, interpolation='nearest', cmap='rainbow')
axs[2].axis('tight')
axs[2].set_title('y asym')
plt.colorbar(im, ax=axs[2])
