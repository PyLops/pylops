r"""
Conjugate gradient preconditioning
==================================
CG can converge slowly when the components of a linear system have very
different scales. Here we solve a weighted smoothing problem,
:math:`(W + L)x = y`, where :math:`W` is a positive diagonal matrix and
:math:`L` is a one-dimensional discrete Laplacian. A Jacobi preconditioner
multiplies each residual component by the reciprocal of the corresponding
diagonal entry of :math:`W + L`.

The preconditioner is passed as an inverse approximation through ``M``. It
changes the search directions while keeping the system and residual cost.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

import pylops
from pylops.optimization.basic import cg

n = 200
grid = np.linspace(0.0, 1.0, n)
weights = np.logspace(0.0, 4.0, n)
A = diags([-np.ones(n - 1), weights + 2.0, -np.ones(n - 1)], [-1, 0, 1])
Aop = pylops.MatrixMult(A)
M = pylops.Diagonal(1.0 / A.diagonal())
x = np.sin(2.0 * np.pi * grid)
y = Aop @ x

xcg, ncg, costcg = cg(Aop, y, niter=4 * n, tol=1e-16)
xpcg, npcg, costpcg = cg(Aop, y, M=M, niter=4 * n, tol=1e-16)
print(f"CG iterations: {ncg}; preconditioned CG iterations: {npcg}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(grid, x, "k", label="Original")
axs[0].plot(grid, xcg, "--", label="CG")
axs[0].plot(grid, xpcg, ":", label="Preconditioned CG")
axs[0].set_xlabel("Position")
axs[0].set_ylabel("Signal")
axs[0].legend()
axs[1].semilogy(costcg / costcg[0], label="CG")
axs[1].semilogy(costpcg / costpcg[0], label="Preconditioned CG")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Relative residual norm")
axs[1].legend()
fig.tight_layout()
