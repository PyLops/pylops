r"""
Linear Regression
=================

This example shows how to use the :py:class:`pylops.LinearRegression` operator
to perform *Linear regression analysis*.

In short, linear regression is the problem of finding the best fitting coefficients,
namely intercept :math:`\mathbf{x_0}` and gradient :math:`\mathbf{x_1}`, for this equation.

    .. math::
        y_i = x_0 + x_1 t_i

As qw can express this problem in a matrix form:

    .. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

our solution can be obtained by solving the following optimization problem:

    .. math::
        J= ||\mathbf{y} - \mathbf{A} \mathbf{x}||_2

See documentation of :py:class:`pylops.LinearRegression` for more detailed definition of
the forward problem.
"""
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

###############################################################################
# Define the input parameters: number of samples along the t-axis (``N``),
# linear regression coefficients (``x``), and standard deviation of noise
# to be added to data (``sigma``).
N = 30
x = np.array([1., 2.])
sigma = 2

###############################################################################
# Let's create the time axis and initialize the :py:class:`pylops.LinearRegression` operator
t = np.arange(N)
LRop = pylops.LinearRegression(t, dtype='float64')

###############################################################################
# We can then apply the operator in forward mode to compute our data points along
# the x-axis (``y``). We will also generate some random gaussian noise and create
# a noisy version of the data (``yn``).
y = LRop*x
yn = y + np.random.normal(0, sigma, N)

###############################################################################
# We are now ready to solve our problem. As we are using an operator from the
# :py:class:`pylops.LinearOperator` family, we can simply use ``/``, which in this case will
# solve the system by means of an iterative solver (i.e., :py:func:`scipy.sparse.linalg.lsqr`).
xest = LRop / y
xnest = LRop / yn

###############################################################################
# Finally let's plot the best fitting line for the case of noise free and noisy data

plt.figure(figsize=(5, 7))
plt.plot(np.array([t.min(), t.max()]),
         np.array([t.min(), t.max()]) * x[1] + x[0], 'k', lw=4,
         label=r'true: $x_0$ = %.2f, $x_1$ = %.2f' % (x[0], x[1]))
plt.plot(np.array([t.min(), t.max()]),
         np.array([t.min(), t.max()]) * xest[1] + xest[0], '--r', lw=4, label='est noise-free')
plt.plot(np.array([t.min(), t.max()]),
         np.array([t.min(), t.max()]) * xnest[1] + xnest[0], '--g', lw=4, label='est noisy')
plt.scatter(t, y, c='r', s=50)
plt.scatter(t, yn, c='g', s=50)
plt.annotate(r'$noise-free: x_0$ = %.2f, $x_1$ = %.2f' % (xest[0], xest[1]),
             xy=(8, 16), xytext=(13, 11), fontsize=8,
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate(r'$noisy: x_0$ = %.2f, $x_1$ = %.2f' % (xest[0], xest[1]),
             xy=(5, 10), xytext=(10, 5), fontsize=8,
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.legend()
