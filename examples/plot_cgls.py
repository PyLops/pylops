r"""
CGLS and LSQR Solvers
=====================

This example shows how to use the :py:func:`pylops.optimization.leastsquares.cgls`
and :py:func:`pylops.optimization.leastsquares.lsqr` PyLops solvers
to minimize the following cost function:

.. math::
        J = \| \mathbf{y} -  \mathbf{Ax} \|_2^2 + \epsilon \| \mathbf{x} \|_2^2

Note that the LSQR solver behaves in the same way as the scipy's
:py:func:`scipy.sparse.linalg.lsqr` solver. However, our solver is also able
to operate on cupy arrays and perform computations on a GPU.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import pylops

plt.close("all")
warnings.filterwarnings("ignore")

###############################################################################
# Let's define a matrix :math:`\mathbf{A}` or size (``N`` and ``M``) and
# fill the matrix with random numbers

N, M = 20, 10
A = np.random.normal(0, 1, (N, M))
Aop = pylops.MatrixMult(A, dtype="float64")

x = np.ones(M)

###############################################################################
# We can now use the cgls solver to invert this matrix

y = Aop * x
xest, istop, nit, r1norm, r2norm, cost_cgls = pylops.optimization.solver.cgls(
    Aop, y, x0=np.zeros_like(x), niter=10, tol=1e-10, show=True
)

print(f"x= {x}")
print(f"cgls solution xest= {xest}")

###############################################################################
# And the lsqr solver to invert this matrix

y = Aop * x
(
    xest,
    istop,
    itn,
    r1norm,
    r2norm,
    anorm,
    acond,
    arnorm,
    xnorm,
    var,
    cost_lsqr,
) = pylops.optimization.solver.lsqr(Aop, y, x0=np.zeros_like(x), niter=10, show=True)

print(f"x= {x}")
print(f"lsqr solution xest= {xest}")


###############################################################################
# Finally we show that the L2 norm of the residual of the two solvers decays
# in the same way, as LSQR is algebrically equivalent to CG on the normal
# equations and CGLS

plt.figure(figsize=(12, 3))
plt.plot(cost_cgls, "k", lw=2, label="CGLS")
plt.plot(cost_lsqr, "--r", lw=2, label="LSQR")
plt.title("Cost functions")
plt.legend()
plt.tight_layout()

###############################################################################
# Note that while we used a dense matrix here, any other linear operator
# can be fed to cgls and lsqr as is the case for any other PyLops solver.
