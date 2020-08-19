import time
import numpy as np

from pylops.utils.backend import get_array_module


def cgls(Op, y, x0, niter=10, damp=0., tol=1e-4,
         show=False, callback=None):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``A`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`np.ndarray`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`np.ndarray`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    damp : :obj:`float`, optional
        Damping coefficient
    tol : :obj:`float`, optional
        Tolerance on residual norm
    show : :obj:`bool`, optional
        Display iterations log
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector

    Returns
    -------
    x : :obj:`np.ndarray`
        Estimated model of size :math:`[M \times 1]`
    iit : :obj:`int`
        Number of executed iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    Notes
    -----
    Minimize the following functional using conjugate gradient iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||_2^2 + \epsilon || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    """
    ncp = get_array_module(x0)

    if show:
        tstart = time.time()
        print('CGLS\n'
              '-----------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'damp = %10e\ttol = %10e\tniter = %d' % (Op.shape[0],
                                                       Op.shape[1],
                                                       damp, tol, niter))
        print(
            '-----------------------------------------------------------------')
        head1 = '    Itn           x[0]              r2norm'
        print(head1)

    if x0 is None:
        x = ncp.zeros(Op.shape[1], dtype=y.dtype)
        s = y.copy()
        r = Op.rmatvec(s)
    else:
        x = x0.copy()
        s = y - Op.matvec(x)
        r = Op.rmatvec(s) - damp * x
    c = r.copy()
    q = Op.matvec(c)
    kold = r.dot(r.conj())

    cost = np.zeros(niter + 1)
    cost[0] = kold + damp * x.dot(x.conj())
    iiter = 0
    while iiter < niter and ncp.abs(kold) > tol:
        a = kold / (q.dot(q.conj()) + damp * c.dot(c.conj()))
        x = x + a * c
        s = s - a * q
        r = Op.rmatvec(s) - damp * x
        k = r.dot(r.conj())
        b = k / kold
        c = r + b * c
        q = Op.matvec(c)
        kold = k
        iiter += 1
        cost[iiter] = kold + damp * x.dot(x.conj())

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                msg = '%6g        %11.4e        %11.4e' % \
                      (iiter, x[0], cost[iiter])
                print(msg)
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (iiter, time.time() - tstart))
        print(
            '-----------------------------------------------------------------\n')
    return x, iiter, cost[:iiter]