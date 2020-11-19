import time
import numpy as np

from pylops.utils.backend import get_array_module


def cg(Op, y, x0, niter=10, damp=0., tol=1e-4, show=False, callback=None):
    r"""Conjugate gradient

    Solve a square system of equations given an operator ``A`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times N]`
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
        Estimated model of size :math:`[N \times 1]`
    iit : :obj:`int`
        Number of executed iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    Notes
    -----
    Solve the :math:`\mathbf{y} = \mathbf{Ax}` problem
    using conjugate gradient iterations.

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
        r = y - Op.matvec(x)
    c = r.copy()
    kold = ncp.abs(r.dot(r.conj()))

    cost = np.zeros(niter + 1)
    cost[0] = kold + damp * ncp.abs(x.dot(x.conj()))
    iiter = 0
    while iiter < niter and kold > tol:
        Opc = Op.matvec(c)
        cOpc = ncp.abs(c.dot(Opc.conj()))
        a = kold / cOpc
        x += a * c
        r -= a * Opc
        k = ncp.abs(r.dot(r.conj()))
        b = k / kold
        c = r + b * c
        kold = k
        iiter += 1
        cost[iiter] = kold

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                if not np.iscomplex(x[0]):
                    msg = '%6g        %11.4e        %11.4e' % \
                          (iiter, x[0], cost[iiter])
                else:
                    msg = '%6g     %4.1e+%4.1ej     %11.4e' % \
                          (iiter, np.real(x[0]), np.imag(x[0]), cost[iiter])
                print(msg)
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (iiter, time.time() - tstart))
        print(
            '-----------------------------------------------------------------\n')
    return x, iiter, cost[:iiter]


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
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{d} = \mathbf{Op}\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    iit : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where
        :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
        \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    Notes
    -----
    Minimize the following functional using conjugate gradient iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

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

    damp = damp ** 2
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
    kold = ncp.abs(r.dot(r.conj()))

    cost = np.zeros(niter + 1)
    cost[0] = kold + damp * ncp.abs(x.dot(x.conj()))
    iiter = 0
    while iiter < niter and kold > tol:
        a = kold / (q.dot(q.conj()) + damp * c.dot(c.conj()))
        x = x + a * c
        s = s - a * q
        r = Op.rmatvec(s) - damp * x
        k = ncp.abs(r.dot(r.conj()))
        b = k / kold
        c = r + b * c
        q = Op.matvec(c)
        kold = k
        iiter += 1
        cost[iiter] = kold + damp * ncp.abs(x.dot(x.conj()))

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                if not np.iscomplex(x[0]):
                    msg = '%6g        %11.4e        %11.4e' % \
                          (iiter, x[0], cost[iiter])
                else:
                    msg = '%6g     %4.1e+%4.1ej     %11.4e' % \
                          (iiter, np.real(x[0]), np.imag(x[0]), cost[iiter])
                print(msg)
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (iiter, time.time() - tstart))
        print(
            '-----------------------------------------------------------------\n')

    # reason for termination
    istop = 1 if kold < tol else 2
    r1norm = kold
    r2norm = cost[iiter]
    return x, istop, iiter, r1norm, r2norm, cost[:iiter]
