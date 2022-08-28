__all__ = [
    "cg",
    "cgls",
    "lsqr",
]

from typing import Callable, List, Optional, Tuple

from pylops import LinearOperator
from pylops.optimization.cls_basic import CG, CGLS, LSQR
from pylops.utils.decorators import add_ndarray_support_to_solver
from pylops.utils.typing import NDArray


@add_ndarray_support_to_solver
def cg(
    Op: LinearOperator,
    y: NDArray,
    x0: Optional[NDArray] = None,
    niter: int = 10,
    tol: float = 1e-4,
    show: bool = False,
    itershow: List[int] = [10, 10, 10],
    callback: Optional[Callable] = None,
) -> Tuple[NDArray, int, NDArray]:
    r"""Conjugate gradient

    Solve a square system of equations given an operator ``Op`` and
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
    tol : :obj:`float`, optional
        Tolerance on residual norm
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
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
        History of the L2 norm of the residual

    Notes
    -----
    See :class:`pylops.optimization.cls_basic.CG`

    """
    cgsolve = CG(Op)
    if callback is not None:
        cgsolve.callback = callback
    x, iiter, cost = cgsolve.solve(
        y=y, x0=x0, tol=tol, niter=niter, show=show, itershow=itershow
    )
    return x, iiter, cost


@add_ndarray_support_to_solver
def cgls(
    Op: LinearOperator,
    y: NDArray,
    x0: Optional[NDArray] = None,
    niter: int = 10,
    damp: float = 0.0,
    tol: float = 1e-4,
    show: bool = False,
    itershow: List[int] = [10, 10, 10],
    callback: Optional[Callable] = None,
) -> Tuple[NDArray, int, int, float, float, NDArray]:
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``Op`` and
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
    itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
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
        :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    iit : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2`, where
        :math:`\mathbf{r} = \mathbf{y} - \mathbf{Op}\,\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
        \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`
    cost : :obj:`numpy.ndarray`, optional
        History of r1norm through iterations

    Notes
    -----
    See :class:`pylops.optimization.cls_basic.CGLS`

    """
    cgsolve = CGLS(Op)
    if callback is not None:
        cgsolve.callback = callback
    x, istop, iiter, r1norm, r2norm, cost = cgsolve.solve(
        y=y, x0=x0, tol=tol, niter=niter, damp=damp, show=show, itershow=itershow
    )
    return x, istop, iiter, r1norm, r2norm, cost


@add_ndarray_support_to_solver
def lsqr(
    Op: LinearOperator,
    y: NDArray,
    x0: Optional[NDArray] = None,
    damp: float = 0.0,
    atol: float = 1e-08,
    btol: float = 1e-08,
    conlim: float = 100000000.0,
    niter: int = 10,
    calc_var: bool = True,
    show: bool = False,
    itershow: List[int] = [10, 10, 10],
    callback: Optional[Callable] = None,
) -> Tuple[NDArray, int, float, float, float, float, float, float, NDArray]:
    r"""LSQR

    Solve an overdetermined system of equations given an operator ``Op`` and
    data ``y`` using LSQR iterations.

    .. math::
      \DeclareMathOperator{\cond}{cond}

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`np.ndarray`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`np.ndarray`, optional
        Initial guess of size :math:`[M \times 1]`
    damp : :obj:`float`, optional
        Damping coefficient
    atol, btol : :obj:`float`, optional
        Stopping tolerances. If both are 1.0e-9, the final residual norm
        should be accurate to about 9 digits. (The solution will usually
        have fewer correct digits, depending on :math:`\cond(\mathbf{Op})`
        and the size of ``damp``.)
    conlim : :obj:`float`, optional
        Stopping tolerance on :math:`\cond(\mathbf{Op})`
        exceeds ``conlim``. For square, ``conlim`` could be as large as 1.0e+12.
        For least-squares problems, ``conlim`` should be less than 1.0e+8.
        Maximum precision can be obtained by setting
        ``atol = btol = conlim = 0``, but the number of iterations may
        then be excessive.
    niter : :obj:`int`, optional
        Number of iterations
    calc_var : :obj:`bool`, optional
        Estimate diagonals of :math:`(\mathbf{Op}^H\mathbf{Op} +
        \epsilon^2\mathbf{I})^{-1}`.
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`list`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector

    Returns
    -------
    x : :obj:`np.ndarray`
        Estimated model of size :math:`[M \times 1]`
    istop : :obj:`int`
        Gives the reason for termination

        ``0`` means the exact solution is :math:`\mathbf{x}=0`

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem

        ``3`` means the estimate of :math:`\cond(\overline{\mathbf{Op}})`
        has exceeded ``conlim``

        ``4`` means :math:`\mathbf{y} - \mathbf{Op}\,\mathbf{x}` is small enough
        for this machine

        ``5`` means the least-squares solution is good enough for this machine

        ``6`` means :math:`\cond(\overline{\mathbf{Op}})` seems to be too large for
        this machine

        ``7`` means the iteration limit has been reached

    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where
        :math:`\mathbf{r} = \mathbf{y} - \mathbf{Op}\,\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
        \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`
    anorm : :obj:`float`
        Estimate of Frobenius norm of :math:`\overline{\mathbf{Op}} =
        [\mathbf{Op} \; \epsilon \mathbf{I}]`
    acond : :obj:`float`
        Estimate of :math:`\cond(\overline{\mathbf{Op}})`
    arnorm : :obj:`float`
        Estimate of norm of :math:`\cond(\mathbf{Op}^H\mathbf{r}-
        \epsilon^2\mathbf{x})`
    var : :obj:`float`
        Diagonals of :math:`(\mathbf{Op}^H\mathbf{Op})^{-1}` (if ``damp=0``)
        or more generally :math:`(\mathbf{Op}^H\mathbf{Op} +
        \epsilon^2\mathbf{I})^{-1}`.
    cost : :obj:`numpy.ndarray`, optional
        History of r1norm through iterations

    Notes
    -----
    See :class:`pylops.optimization.cls_basic.LSQR`

    """
    lsqrsolve = LSQR(Op)
    if callback is not None:
        lsqrsolve.callback = callback
    (
        x,
        istop,
        iiter,
        r1norm,
        r2norm,
        anorm,
        acond,
        arnorm,
        xnorm,
        var,
        cost,
    ) = lsqrsolve.solve(
        y=y,
        x0=x0,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        niter=niter,
        calc_var=calc_var,
        show=show,
        itershow=itershow,
    )
    return x, istop, iiter, r1norm, r2norm, anorm, acond, arnorm, xnorm, var, cost
