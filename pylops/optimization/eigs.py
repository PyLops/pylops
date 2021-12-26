import numpy as np

from pylops.utils.backend import get_module


def power_iteration(Op, niter=10, tol=1e-5, dtype="float32", backend="numpy"):
    """Power iteration algorithm.

    Power iteration algorithm, used to compute the largest eigenvector and
    corresponding eigenvalue. Note that for complex numbers, the eigenvalue
    with largest module is found.

    This implementation closely follow that of
    https://en.wikipedia.org/wiki/Power_iteration.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Square operator
    niter : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Update tolerance
    dtype : :obj:`str`, optional
        Type of elements in input array.
    backend : :obj:`str`, optional
        Backend to use (`numpy` or `cupy`)

    Returns
    -------
    maxeig : :obj:`int`
        Largest eigenvalue
    b_k : :obj:`np.ndarray` or :obj:`cp.ndarray`
        Largest eigenvector
    iiter : :obj:`int`
        Effective number of iterations

    """
    ncp = get_module(backend)

    # Identify if operator is complex
    if np.issubdtype(dtype, np.complexfloating):
        cmpx = 1j
    else:
        cmpx = 0

    # Choose a random vector to decrease the chance that vector
    # is orthogonal to the eigenvector
    b_k = ncp.random.rand(Op.shape[1]).astype(dtype) + cmpx * ncp.random.rand(
        Op.shape[1]
    ).astype(dtype)
    b_k = b_k / ncp.linalg.norm(b_k)

    niter = 10 if niter is None else niter
    maxeig_old = 0.0
    for iiter in range(niter):
        # compute largest eigenvector
        b1_k = Op.matvec(b_k)

        # compute largest eigevalue
        maxeig = ncp.vdot(b_k, b1_k)

        # renormalize the vector
        b_k = b1_k / ncp.linalg.norm(b1_k)

        if ncp.abs(maxeig - maxeig_old) < tol * maxeig_old:
            break
        maxeig_old = maxeig

    return maxeig, b_k, iiter + 1
