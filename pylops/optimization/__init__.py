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

    NormalEquationsInversion        Inversion of normal equations.
    RegularizedInversion            Regularized inversion.
    PreconditionedInversion         Preconditioned inversion.

and sparsity-promoting solvers in pylops.optimization.sparsity:

    IRLS	                        Iteratively reweighted least squares.
    OMP	                            Orthogonal Matching Pursuit (OMP).
    ISTA                            Iterative Soft Thresholding Algorithm.
    FISTA                           Fast Iterative Soft Thresholding Algorithm.
    SPGL1                           Spectral Projected-Gradient for L1 norm.
    SplitBregman	                Split Bregman for mixed L2-L1 norms.

"""
