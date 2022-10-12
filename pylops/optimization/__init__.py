"""
Optimization
============

The subpackage optimization provides an extensive set of solvers to be
used with PyLops linear operators.

A list of least-squares solvers in pylops.optimization.solver:

    cg                              Conjugate gradient.
    cgls                            Conjugate gradient least-squares.
    lsqr                            LSQR.

and wrappers for regularized or preconditioned inversion in pylops.optimization.leastsquares:

    normal_equations_inversion       Inversion of normal equations.
    regularized_inversion            Regularized inversion.
    preconditioned_inversion         Preconditioned inversion.

and sparsity-promoting solvers in pylops.optimization.sparsity:

    irls                             Iteratively reweighted least squares.
    omp	                             Orthogonal Matching Pursuit (OMP).
    ista                             Iterative Soft Thresholding Algorithm.
    fista                            Fast Iterative Soft Thresholding Algorithm.
    spgl1                            Spectral Projected-Gradient for L1 norm.
    splitbregman                     Split Bregman for mixed L2-L1 norms.

Note that these solvers are thin wrappers over class-based solvers (new in v2), which can be accessed from
submodules with equivalent name and suffix c.

"""
