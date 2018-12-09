r"""
Matrix Multiplication
=====================

This example shows how to use the :py:class:`pylops.MatrixMult` operator
to perform *Matrix inversion* of the following linear system.

.. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

You will see that since this operator is a simple overloading to a
:py:func:`numpy.ndarray` object, the solution of the linear system
can be obtained via both direct inversion (i.e., by means explicit
solver such as :py:func:`scipy.linalg.solve` or :py:func:`scipy.linalg.lstsq`)
and iterative solver (i.e., :py:func:`from scipy.sparse.linalg.lsqr`).

Note that in case of rectangular :math:`\mathbf{A}`, an exact inverse does
not exist and a least-square solution is computed instead.
"""

import numpy as np
from scipy.sparse.linalg import lsqr

import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs

import pylops

plt.close('all')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Let's define the size of matrix :math:`\mathbf{A}` (``N`` and ``M``) and
# fill matrix with random numbers

N, M = 20, 20
A = np.random.normal(0, 1, (N, M))
x = np.ones(M)

#a = 1
Aop = pylops.MatrixMult(A, dtype='float64')

###############################################################################
# We can now apply the forward operator to create the data vector :math:`\mathbf{y}`
# and use ``/`` to solve the system by means of an explicit solver.

y = Aop*x
xest = Aop/y

###############################################################################
# Let's visually plot the system of equations we just solved.
gs = pltgs.GridSpec(1, 6)
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(gs[0, 0])
ax.imshow(y[:, np.newaxis], cmap='rainbow')
ax.set_title('y', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 1])
ax.text(0.35, 0.5, '=', horizontalalignment='center',
        verticalalignment='center', size=40, fontweight='bold')
ax.axis('off')
ax = plt.subplot(gs[0, 2:5])
ax.imshow(Aop.A, cmap='rainbow')
ax.set_title('A', size=20, fontweight='bold')
ax.set_xticks(np.arange(N-1)+0.5)
ax.set_yticks(np.arange(M-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 5])
ax.imshow(x[:, np.newaxis], cmap='rainbow')
ax.set_title('x', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(M-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

gs = pltgs.GridSpec(1, 6)
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(gs[0, 0])
ax.imshow(x[:, np.newaxis], cmap='rainbow')
ax.set_title('xest', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(M-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 1])
ax.text(0.35, 0.5, '=', horizontalalignment='center',
        verticalalignment='center', size=40, fontweight='bold')
ax.axis('off')
ax = plt.subplot(gs[0, 2:5])
ax.imshow(Aop.inv(), cmap='rainbow')
ax.set_title(r'A$^{-1}$', size=20, fontweight='bold')
ax.set_xticks(np.arange(N-1)+0.5)
ax.set_yticks(np.arange(M-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 5])
ax.imshow(y[:, np.newaxis], cmap='rainbow')
ax.set_title('y', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

###############################################################################
# Let's also plot the matrix eigenvalues

plt.figure(figsize=(8, 3))
plt.plot(Aop.eigs(), 'k', lw=2)
plt.title('Eigenvalues', size=20, fontweight='bold')
plt.xlabel('#eigenvalue')
plt.xlabel('intensity')
plt.tight_layout()

###############################################################################
# We can also repeat the same exercise for a non-square matrix
N, M = 200, 50
A = np.random.normal(0, 1, (N, M))
x = np.ones(M)

Aop = pylops.MatrixMult(A, dtype='float64')
y = Aop*x
yn = y + np.random.normal(0, 0.3, N)

xest = Aop/y
xnest = Aop/yn

plt.figure(figsize=(8, 3))
plt.plot(x, 'k', lw=2, label='True')
plt.plot(xest, '--r', lw=2, label='Noise-free')
plt.plot(xnest, '--g', lw=2, label='Noisy')
plt.title('Matrix inversion', size=20, fontweight='bold')
plt.legend()

###############################################################################
# Finally, the same matrix can be applied to multiple columns of of input model
# :math:`\mathbf{x}`. We can express this operation in a matrix form:
#
#    .. math::
#       \mathbf{y} =
#       \begin{bmatrix}
#               \mathbf{A}  \quad \mathbf{0}  \quad  \mathbf{0}  \\
#               \mathbf{0}  \quad \mathbf{A}  \quad  \mathbf{0}  \\
#               \mathbf{0}  \quad \mathbf{0}  \quad  \mathbf{A}
#               \end{bmatrix}
#       \begin{bmatrix}
#               \mathbf{x_1}  \\
#               \mathbf{x_2}  \\
#               \mathbf{x_3}
#       \end{bmatrix}

A = np.array([[1., 2.], [4., 5.]])
x = np.array([[1., 1.], [2., 2.], [3., 3.]]).flatten()

Aop = pylops.MatrixMult(A, dims=(3), dtype='float64')
y = Aop*x


xest, istop, itn, r1norm, r2norm = \
        lsqr(Aop, y, damp=1e-10, iter_lim=10, show=0)[0:5]

print('A= %s' % A)
print('x= %s' % x)
print('lsqr solution xest= %s' % xest)
