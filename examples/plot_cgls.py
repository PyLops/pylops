r"""
CGLS Solver
===========

This example shows how to use the :py:func:`pylops.optimization.leastsquares.cgls`
solver to minimize the following cost function:

.. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||_2^2 + \epsilon || \mathbf{x} ||_2^2

"""

import warnings
import numpy as np
from scipy.sparse import rand
from scipy.sparse.linalg import lsqr

import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs

import pylops

plt.close('all')
warnings.filterwarnings('ignore')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Let's define a matrix :math:`\mathbf{A}` or size (``N`` and ``M``) and
# fill the matrix with random numbers

N, M = 20, 10
A = np.random.normal(0, 1, (N, M))
Aop = pylops.MatrixMult(A, dtype='float64')

x = np.ones(M)

###############################################################################
# We can now use the cgls solver to invert this matrix

y = Aop * x
xest, nit, cost = \
    pylops.optimization.leastsquares.cgls(Aop, y, x0=np.zeros_like(x),
                                          niter=10, tol=1e-10, show=True)


print('x= %s' % x)
print('cgls solution xest= %s' % xest)

plt.figure(figsize=(12, 3))
plt.plot(cost, 'k', lw=2)
plt.title('Cost function')

###############################################################################
# Note that while we used a dense matrix here, any other linear operator
# can be fed to cgls as is the case for any other PyLops solver.



