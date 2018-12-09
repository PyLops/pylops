"""
Operators concatenation
=======================

This example shows how to use stacking operators such as :py:class:`pylops.VStack`,
:py:class:`pylops.HStack`, and  :py:class:`pylops.BlockDiag`.

These operators allow for different combinations of multiple linear operators in
a single operator. Such functionalities are used within PyLops as the basis for
the creation of complex operators as well as in the definition of various types
of optimization problems with regularization or preceonditioning.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Let's start by defining two second derivatives :py:class:`pylops.SecondDerivative`
# that we will be using in this example.
D2hop = pylops.SecondDerivative(11 * 21, dims=[11, 21], dir=1, dtype='float32')
D2vop = pylops.SecondDerivative(11 * 21, dims=[11, 21], dir=0, dtype='float32')

###############################################################################
# Chaining of operators represents the simplest concatenation that
# can be performed between two or more linear operators.
# This can be easily achieved using the ``*`` symbol
#
#    .. math::
#       \mathbf{D_{cat}}=  \mathbf{D_v} \mathbf{D_h}
Nv, Nh = 11, 21
X = np.zeros((Nv, Nh))
X[int(Nv/2), int(Nh/2)] = 1

D2op = D2vop*D2hop
Y = np.reshape(D2op*X.flatten(), (Nv, Nh))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Chain', fontsize=14, fontweight='bold')
im = axs[0].imshow(X, interpolation='nearest')
axs[0].axis('tight')
axs[0].set_title(r'$x$')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Y, interpolation='nearest')
axs[1].axis('tight')
axs[1].set_title(r'$y=(D_x+D_y) x$')
plt.colorbar(im, ax=axs[1])
plt.tight_layout()

###############################################################################
# We now want to *vertically stack* three operators
#
#    .. math::
#       \mathbf{D_{Vstack}} =
#        \begin{bmatrix}
#          \mathbf{D_v}    \\
#          \mathbf{D_h}
#        \end{bmatrix}, \qquad
#       \mathbf{y} =
#        \begin{bmatrix}
#          \mathbf{D_v}\mathbf{x}    \\
#          \mathbf{D_h}\mathbf{x}
#        \end{bmatrix}

Nv, Nh = 11, 21
X = np.zeros((Nv, Nh))
X[int(Nv/2), int(Nh/2)] = 1
Dstack = pylops.VStack([D2vop, D2hop])

Y = np.reshape(Dstack * X.flatten(), (Nv * 2, Nh))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Vertical stacking', fontsize=14, fontweight='bold')
im = axs[0].imshow(X, interpolation='nearest')
axs[0].axis('tight')
axs[0].set_title(r'$x$')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Y, interpolation='nearest')
axs[1].axis('tight')
axs[1].set_title(r'$y$')
plt.colorbar(im, ax=axs[1])
plt.tight_layout()

###############################################################################
# Similarly we can now *horizontally stack* three operators
#
#    .. math::
#       \mathbf{D_{Hstack}} =
#        \begin{bmatrix}
#           \mathbf{D_v}  & 0.5*\mathbf{D_v} & -1*\mathbf{D_h}
#        \end{bmatrix}, \qquad
#       \mathbf{y} =
#        \mathbf{D_v}\mathbf{x}_1 + \mathbf{D_h}\mathbf{x}_2

Nv, Nh = 11, 21
X = np.zeros((Nv*3, Nh))
X[int(Nv/2), int(Nh/2)] = 1
X[int(Nv/2) + Nv, int(Nh/2)] = 1
X[int(Nv/2) + 2*Nv, int(Nh/2)] = 1

Hstackop = pylops.HStack([D2vop, 0.5 * D2vop, -1 * D2hop])
Y = np.reshape(Hstackop*X.flatten(), (Nv, Nh))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Horizontal stacking', fontsize=14, fontweight='bold')
im = axs[0].imshow(X, interpolation='nearest')
axs[0].axis('tight')
axs[0].set_title(r'$x$')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Y, interpolation='nearest')
axs[1].axis('tight')
axs[1].set_title(r'$y$')
plt.colorbar(im, ax=axs[1])
plt.tight_layout()

###############################################################################
# Finally we can use the *block-diagonal operator* to apply three operators
# to three different subset of the model and data
#
#    .. math::
#       \mathbf{D_{BDiag}} =
#        \begin{bmatrix}
#           \mathbf{D_v}  & \mathbf{0}       &  \mathbf{0}  \\
#           \mathbf{0}    & 0.5*\mathbf{D_v} &  \mathbf{0}  \\
#           \mathbf{0}    & \mathbf{0}       &  -\mathbf{D_h}
#        \end{bmatrix}, \qquad
#       \mathbf{y} =
#        \begin{bmatrix}
#           \mathbf{D_v}     \mathbf{x_1}  \\
#           0.5*\mathbf{D_v} \mathbf{x_2}  \\
#           -\mathbf{D_h}  \mathbf{x_3}
#        \end{bmatrix}

Nv, Nh = 11, 21
X = np.zeros((Nv*3, Nh))
X[int(Nv/2), int(Nh/2)] = 1
X[int(Nv/2) + Nv, int(Nh/2)] = 1
X[int(Nv/2) + 2*Nv, int(Nh/2)] = 1

Block = pylops.BlockDiag([D2vop, 0.5 * D2vop, -1 * D2hop])
Y = np.reshape(Block*np.ndarray.flatten(X), (11*3, 21))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('Block-diagonal', fontsize=14, fontweight='bold')
im = axs[0].imshow(X, interpolation='nearest')
axs[0].axis('tight')
axs[0].set_title(r'$x$')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Y, interpolation='nearest')
axs[1].axis('tight')
axs[1].set_title(r'$y$')
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
