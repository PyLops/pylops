"""
Optimization
============
This tutorial will guide you through the :py:mod:`pylops.optimization` module and
discuss various options for solving systems of linear equations constructed by
means of PyLops linear operators

"""
# pylint: disable=C0103
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# To start let's consider the simplest problem, *least-square inversion without regularization*.
# By expressing the forward problem in a matrix form :math:`\mathbf{y}= \mathbf{A} \mathbf{x}`,
# we aim here to minimize the following cost function:
#
#   .. math::
#        J= ||\mathbf{y} - \mathbf{A} \mathbf{x}||_2
#
# Depending on the choice of the operator :math:`\mathbf{A}`, such problem can be solved using
# explicit matrix solvers as well as iterative solvers.
#
# In this case we will trying to solve a *data reconstruction* problem. and we be using the
# latter approach (more specifically the scipy implementation of *LSQR* solver - i.e.,
# :py:func:`scipy.sparse.linalg.lsqr`) as we do not want to explicitly create and invert a matrix.
# In most cases this will be the only viable approach as most of the large-scale
# optimization problems that we are interested to solve using PyLops do not lend
# naturally to the creation and inversion of explicit matrices.
#
# We will also be considering two different starting guesses:
#
# * starting guess equal 0
# * arbitrary choice of starting guess
#
# First, let's start by defining and creating the forward problem. Consider the problem
# of reconstructing a regularly sampled signal of size :math:`M` from :math:`N`
# randomly selected samples:
#   .. math::
#       \mathbf{y} = \mathbf{R} \mathbf{x}
#
# where the restriction operator :math:`\mathbf{R}` that selects the :math:`M` elements from
# :math:`\mathbf{x}` at random locations is implemented using :py:class:`pylops.Restriction`, and
#
#   .. math::
#       \mathbf{y}= [y_1, y_2,...,y_N]^T, \qquad \mathbf{x}= [x_1, x_2,...,x_M]^T, \qquad
#
# with :math:`M>>N`.

# Signal creation
np.random.seed(seed=4)
freqs = (5., 3., 8.)
amps = (1., 1., 1.)
N = 200
dt = 0.004
t = np.arange(N)*dt
x = np.zeros(N)

for freq, amp in zip(freqs, amps):
    x = x + amp*np.sin(2*np.pi*freq*t)

# subsampling locations
perc_subsampling = 0.4
Nsub = int(np.round(N*perc_subsampling))

iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])

# Create restriction operator
Rop = pylops.Restriction(N, iava, dtype='float64')

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
# Back to our first cost function, this can be very easily implemented using the
# :math:`/` for PyLops operators, which will automatically call the
# :py:func:`scipy.sparse.linalg.lsqr` with some default parameters.
xinv = Rop / y

###############################################################################
# We could also use :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`
# (without regularization term for now) and customize our solvers using ``kwargs``.
xinv = \
    pylops.optimization.leastsquares.RegularizedInversion(Rop, [], y,
                                                          **dict(damp=0, iter_lim=10, show=1))

###############################################################################
# And finally select a different starting guess from the null vector
xinv_fromx0 = \
    pylops.optimization.leastsquares.RegularizedInversion(Rop, [], y,
                                                          x0=np.ones(N),
                                                          **dict(damp=0, iter_lim=10, show=0))

###############################################################################
# The cost function above can be also expanded in terms of its *normal equations*
#
#   .. math::
#       \mathbf{x}_{ne}= (\mathbf{R}^H \mathbf{R}^H)^{-1} \mathbf{R}^H \mathbf{y}
#
# The method :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion` implements
# such system of equations explicitly and solves them using an iterative scheme suitable
# for square matrices (i.e., :math:`M=N`). While this approach may seem not very useful,
# we will soon see how regularization terms could be easily added to the
# normal equations using this method.

xne = pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [], y)

###############################################################################
# Let's visualize the different inversion results
fig = plt.figure(figsize=(15, 5))
plt.plot(t, x, 'k', lw=2, label='original')
plt.plot(t, xinv, 'b', ms=10, label='inversion')
plt.plot(t, xinv_fromx0, '--r', ms=10, label='inversion from x0')
plt.plot(t, xne, '--g', ms=10, label='normal equations')
plt.legend()
plt.title('Data reconstruction without regularization')


###############################################################################
# Regularization
# ~~~~~~~~~~~~~~
# You may have noticed that none of the inversion has been successfull in recovering
# the original signal. This is a clear indication that the problem we are trying to
# solve is highly ill-posed and requires some prior knowledge from the user.
#
# We will now see how to add prior information to the inverse process in the
# form of regularization (or preconditioning). This can be done in two different ways
#
# * regularization via :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion` or
#   :py:func:`pylops.optimization.leastsquares.RegularizedInversion`)
# * preconditioning via :py:func:`pylops.optimization.leastsquares.PreconditionedInversion`
#
# Let's start by regularizing the normal equations using a second derivative operator
#
#   .. math::
#       \mathbf{x} = (\mathbf{R^TR}+\epsilon_\nabla^2\nabla^T\nabla)^{-1} \mathbf{R^Ty}

# Create regularization operator
D2op = pylops.SecondDerivative(N, dims=None, dtype='float64')

# Regularized inversion
epsR = np.sqrt(0.1)
epsI = np.sqrt(1e-4)

xne = pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [D2op], y, epsI=epsI,
                                                                epsRs=[epsR], returninfo=False,
                                                                **dict(maxiter=50))

###############################################################################
# We can do the same while using :py:func:`pylops.optimization.leastsquares.RegularizedInversion`
# which solves the following augmented problem
#
#   .. math::
#       \begin{bmatrix}
#           \mathbf{R}    \\
#           \epsilon_\nabla \nabla
#       \end{bmatrix} \mathbf{x} =
#           \begin{bmatrix}
#           \mathbf{y}    \\
#           0
#       \end{bmatrix}

xreg = pylops.optimization.leastsquares.RegularizedInversion(Rop, [D2op], y,
                                                             epsRs=[np.sqrt(0.1)], returninfo=False,
                                                             **dict(damp=np.sqrt(1e-4),
                                                                  iter_lim=50, show=0))

###############################################################################
# We can also write a preconditioned problem, whose cost function is
#
#   .. math::
#       J= ||\mathbf{y} - \mathbf{R} \mathbf{P} \mathbf{p}||_2
#
# where :math:`\mathbf{P}` is the precondioned operator, :math:`\mathbf{p}` is the projected model
# in the preconditioned space, and :math:`\mathbf{x}=\mathbf{P}\mathbf{p}`is the model in
# the original model space we want to solve for. Note that a preconditioned problem converges
# much faster to its solution than its corresponding regularized problem. This can be done
# using the routine :py:func:`pylops.optimization.leastsquares.PreconditionedInversion`.

# Create regularization operator
Sop = pylops.Smoothing1D(nsmooth=11, dims=[N], dtype='float64')

# Invert for interpolated signal
xprec = pylops.optimization.leastsquares.PreconditionedInversion(Rop, Sop, y,
                                                                 returninfo=False,
                                                                 **dict(damp=np.sqrt(1e-9),
                                                                      iter_lim=20, show=0))

###############################################################################
# Let's finally visualize these solutions

# sphinx_gallery_thumbnail_number=3
fig = plt.figure(figsize=(15, 5))
plt.plot(t[iava], y, '.k', ms=20, label='available samples')
plt.plot(t, x, 'k', lw=3, label='original')
plt.plot(t, xne, 'b', lw=3, label='normal equations')
plt.plot(t, xreg, '--r', lw=3, label='regularized')
plt.plot(t, xprec, '--g', lw=3, label='preconditioned equations')
plt.legend()
plt.title('Data reconstruction with regularization')

subax = fig.add_axes([0.7, 0.2, 0.15, 0.6])
subax.plot(t[iava], y, '.k', ms=20)
subax.plot(t, x, 'k', lw=3)
subax.plot(t, xne, 'b', lw=3)
subax.plot(t, xreg, '--r', lw=3)
subax.plot(t, xprec, '--g', lw=3)
subax.set_xlim(0.05, 0.3)

plt.show()

###############################################################################
# Much better estimates! We have seen here how regularization and/or preconditioning
# can be vital to succesfully solve some ill-posed inverse problems.
