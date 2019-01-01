"""
1. The LinearOpeator
====================
This first tutorials is aimed at easing the use of the PyLops
library for both new users and developers.

Since PyLops heavily relies  on the use of the
:py:class:`scipy.sparse.linalg.LinearOperator` class of scipy, we will start
by looking at how to initialize a linear operator as well as
different ways to apply the forward and adjoint operations. Finally we will
investigate various *special methods*, also called *magic methods*
(i.e., methods with the double underscores at the beginning and the end) that
have been implemented for such a class and will allow summing, subtractring,
chaining, etc. multiple operators in very easy and expressive way.
"""

###############################################################################
# Let's start by defining a simple operator that applies element-wise
# multiplication of the model with a vector ``d`` in forward mode and
# element-wise multiplication of the data with the same vector ``d`` in
# adjoint mode. This operator is present in PyLops under the
# name of :py:class:`pylops.Diagonal` and
# its implementation is discussed in more details in the :ref:`addingoperator`
# page.
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pylops

n = 10
d = np.arange(n) + 1.
x = np.ones(n)
Dop = pylops.Diagonal(d)

###############################################################################
# First of all we apply the operator in the forward mode. This can be done in
# four different ways:
#
# * ``_matvec``: directly applies the method implemented for forward mode
# * ``matvec``: performs some checks before and after applying ``_matvec``
# * ``*``: operator used to map the special method ``__matmul__`` which
#   checks whether the input ``x`` is a vector or matrix and applies ``_matvec``
#   or ``_matmul`` accordingly.
# * ``@``: operator used to map the special method ``__mul__`` which
#   performs like the ``*`` opetator
#
# We will time these 4 different executions and see how using ``_matvec``
# (or ``matvec``) will result in the faster computation. It is thus advised to
# use ``*`` (or ``@``) in examples when expressivity has priority but prefer
# ``_matvec`` (or ``matvec``) for efficient implementations.

# setup command
cmd_setup ="""\
import numpy as np
import pylops
n = 10
d = np.arange(n) + 1.
x = np.ones(n)
Dop = pylops.Diagonal(d)
DopH = Dop.H
"""

# _matvec
cmd1 = 'Dop._matvec(x)'

# matvec
cmd2 = 'Dop.matvec(x)'

# @
cmd3 = 'Dop@x'

# *
cmd4 = 'Dop*x'

# timing
t1 = 1.e3 * np.array(timeit.repeat(cmd1, setup=cmd_setup,
                                   number=500, repeat=5))
t2 = 1.e3 * np.array(timeit.repeat(cmd2, setup=cmd_setup,
                                   number=500, repeat=5))
t3 = 1.e3 * np.array(timeit.repeat(cmd3, setup=cmd_setup,
                                   number=500, repeat=5))
t4 = 1.e3 * np.array(timeit.repeat(cmd4, setup=cmd_setup,
                                   number=500, repeat=5))

plt.figure(figsize=(7, 2))
plt.plot(t1, 'k', label=' _matvec')
plt.plot(t2, 'r', label='matvec')
plt.plot(t3, 'g', label='@')
plt.plot(t4, 'b', label='*')
plt.legend()
plt.axis('tight')

###############################################################################
# Similarly we now consider the adjoint mode. This can be done in
# three different ways:
#
# * ``_rmatvec``: directly applies the method implemented for adjoint mode
# * ``rmatvec``: performs some checks before and after applying ``_rmatvec``
# * ``.H*``: first applies the adjoint ``.H`` which creates a new
#   `scipy.sparse.linalg._CustomLinearOperator`` where ``_matvec``
#   and ``_rmatvec`` are swapped and then applies the new ``_matvec``.
#
# Once again, after timing these 3 different executions we can see
# see how using ``_rmatvec`` (or ``rmatvec``) will result in the faster
# computation while ``.H*`` is very unefficient and slow. Note that if the
# adjoint has to be applied multiple times it is at least advised to create
# the adjoint operator by applying ``.H`` only once upfront.
# Not surprisingly, the linear solvers in scipy as well as in PyLops
# actually use ``matvec`` and ``rmatvec`` when dealing with linear operators.

# _rmatvec
cmd1 = 'Dop._rmatvec(x)'

# rmatvec
cmd2 = 'Dop.rmatvec(x)'

# .H* (pre-computed H)
cmd3 = 'DopH*x'

# .H*
cmd4 = 'Dop.H*x'

# timing
t1 = 1.e3 * np.array(timeit.repeat(cmd1, setup=cmd_setup,
                                   number=500, repeat=5))
t2 = 1.e3 * np.array(timeit.repeat(cmd2, setup=cmd_setup,
                                   number=500, repeat=5))
t3 = 1.e3 * np.array(timeit.repeat(cmd3, setup=cmd_setup,
                                   number=500, repeat=5))
t4 = 1.e3 * np.array(timeit.repeat(cmd4, setup=cmd_setup,
                                   number=500, repeat=5))

plt.figure(figsize=(7, 2))
plt.plot(t1, 'k', label=' _rmatvec')
plt.plot(t2, 'r', label='rmatvec')
plt.plot(t3, 'g', label='.H* (pre-computed H)')
plt.plot(t4, 'b', label='.H*')
plt.legend()
plt.axis('tight')

###############################################################################
# Just to reiterate once again, it is advised to call ``matvec``
# and ``rmatvec`` unless PyLops linear operators are used for
# teaching purposes.
#
# Finally we go through some other *methods* and *special methods* that
# are implemented in :py:class:`scipy.sparse.linalg.LinearOperator` (and
# :py:class:`pylops.LinearOperator`):
#
# * ``Op1+Op2``: maps the special method ``__add__`` and
#   performs summation between two operators
# * ``-Op``: maps the special method ``__neg__`` and
#   performs negation of an operators
# * ``Op1-Op2``: maps the special method ``__sub__`` and
#   performs summation between two operators
# * ``Op1**N``: maps the special method ``__pow__`` and
#   performs exponentiation of an operator
# * ``Op/y`` (and ``Op.div(y)``): maps the special method ``__truediv__`` and
#   performs inversion of an operator
# * ``Op.eigs()``: estimates the eigenvalues of the operator
# * ``Op.cond()``: estimates the condition number of the operator

# +
print(Dop+Dop)

# -
print(-Dop)
print(Dop-0.5*Dop)

# **
print(Dop**3)

#* and /
y = Dop*x
print(Dop/y)

# eigs
print(Dop.eigs(neigs=3))

# cond
print(Dop.cond())

###############################################################################
# This first tutorial is completed. You have seen the basic operations that
# can be performed using :py:class:`scipy.sparse.linalg.LinearOperator` and
# our overload of such a class :py:class:`pylops.LinearOperator` and you
# should be able to get started combining various PyLops operators and
# solving your own inverse problems.
