from pylops.optimization.cls_leastsquares import (
    NormalEquationsInversion,
    PreconditionedInversion,
    RegularizedInversion,
)


def normal_equations_inversion(
    Op,
    y,
    Regs,
    x0=None,
    Weight=None,
    dataregs=None,
    epsI=0,
    epsRs=None,
    NRegs=None,
    epsNRs=None,
    engine="scipy",
    show=False,
    **kwargs_solver,
):
    r"""Inversion of normal equations.

    Solve the regularized normal equations for a system of equations
    given the operator ``Op``, a data weighting operator ``Weight`` and
    optionally a list of regularization terms ``Regs`` and/or ``NRegs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`numpy.ndarray`
        Data of size :math:`[N \times 1]`
    Regs : :obj:`list`
        Regularization operators (``None`` to avoid adding regularization)
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess of size :math:`[M \times 1]`
    Weight : :obj:`pylops.LinearOperator`, optional
        Weight operator
    dataregs : :obj:`list`, optional
        Regularization data (must have the same number of elements
        as ``Regs``)
    epsI : :obj:`float`, optional
        Tikhonov damping
    epsRs : :obj:`list`, optional
         Regularization dampings (must have the same number of elements
         as ``Regs``)
    NRegs : :obj:`list`
        Normal regularization operators (``None`` to avoid adding
        regularization). Such operators must apply the chain of the
        forward and the adjoint in one go. This can be convenient in
        cases where a faster implementation is available compared to applying
        the forward followed by the adjoint.
    epsNRs : :obj:`list`, optional
         Regularization dampings for normal operators (must have the same
         number of elements as ``NRegs``)
    engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
    show : :obj:`bool`, optional
         Display normal equations solver log
    **kwargs_solver
        Arbitrary keyword arguments for chosen solver
        (:py:func:`scipy.sparse.linalg.cg` and
        :py:func:`pylops.optimization.solver.cg` are used for engine ``scipy``
        and ``pylops``, respectively)

        .. note::
            When user does not supply ``atol``, it is set to "legacy".

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Convergence information (only when using :py:func:`scipy.sparse.linalg.cg`):

        ``0``: successful exit

        ``>0``: convergence to tolerance not achieved, number of iterations

        ``<0``: illegal input or breakdown

    See Also
    --------
    RegularizedInversion: Regularized inversion
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    See :class:`pylops.optimization.leastsquares.NormalEquationsInversion`

    """
    nesolve = NormalEquationsInversion(Op)
    xinv, istop = nesolve.solve(
        y,
        Regs,
        x0=x0,
        Weight=Weight,
        dataregs=dataregs,
        epsI=epsI,
        epsRs=epsRs,
        NRegs=NRegs,
        epsNRs=epsNRs,
        engine=engine,
        show=show,
        **kwargs_solver,
    )
    return xinv, istop


def regularized_inversion(
    Op,
    y,
    Regs,
    x0=None,
    Weight=None,
    dataregs=None,
    epsRs=None,
    engine="scipy",
    show=False,
    **kwargs_solver,
):
    r"""Regularized inversion.

    Solve a system of regularized equations given the operator ``Op``,
    a data weighting operator ``Weight``, and a list of regularization
    terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`numpy.ndarray`
        Data of size :math:`[N \times 1]`
    Regs : :obj:`list`
        Regularization operators (``None`` to avoid adding regularization)
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess of size :math:`[M \times 1]`
    Weight : :obj:`pylops.LinearOperator`, optional
        Weight operator
    dataregs : :obj:`list`, optional
        Regularization data (if ``None`` a zero data will be used for every
        regularization operator in ``Regs``)
    epsRs : :obj:`list`, optional
         Regularization dampings
    engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
    show : :obj:`bool`, optional
         Display normal equations solver log
    **kwargs_solver
        Arbitrary keyword arguments for chosen solver
        (:py:func:`scipy.sparse.linalg.lsqr` and
        :py:func:`pylops.optimization.solver.cgls` are used for engine ``scipy``
        and ``pylops``, respectively)

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where
        :math:`\mathbf{r} = \mathbf{y} - \mathbf{Op}\,\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
        \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`

    See Also
    --------
    RegularizedOperator: Regularized operator
    NormalEquationsInversion: Normal equations inversion
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    See :class:`pylops.optimization.leastsquares.RegularizedInversion`

    """
    rsolve = RegularizedInversion(Op)
    xinv, istop, itn, r1norm, r2norm = rsolve.solve(
        y,
        Regs,
        x0=x0,
        Weight=Weight,
        dataregs=dataregs,
        epsRs=epsRs,
        engine=engine,
        show=show,
        **kwargs_solver,
    )
    return xinv, istop, itn, r1norm, r2norm


def preconditioned_inversion(
    Op, y, P, x0=None, engine="scipy", show=False, **kwargs_solver
):
    r"""Preconditioned inversion.

    Solve a system of preconditioned equations given the operator
    ``Op`` and a preconditioner ``P``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`numpy.ndarray`
        Data of size :math:`[N \times 1]`
    P : :obj:`pylops.LinearOperator`
        Preconditioner
    x0 : :obj:`numpy.ndarray`
        Initial guess of size :math:`[M \times 1]`
    engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
    show : :obj:`bool`, optional
         Display normal equations solver log
    **kwargs_solver
        Arbitrary keyword arguments for chosen solver
        (:py:func:`scipy.sparse.linalg.lsqr` and
        :py:func:`pylops.optimization.solver.cgls` are used as default for numpy
        and cupy `data`, respectively)

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where :math:`\mathbf{r} = \mathbf{y} -
        \mathbf{Op}\,\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +  \epsilon^2
        \mathbf{x}^T\mathbf{x}}`. Equal to ``r1norm`` if :math:`\epsilon=0`

    See Also
    --------
    RegularizedInversion: Regularized inversion
    NormalEquationsInversion: Normal equations inversion

    Notes
    -----
    See :class:`pylops.optimization.leastsquares.PreconditionedInversion`

    """
    psolve = PreconditionedInversion(Op)
    xinv, istop, itn, r1norm, r2norm = psolve.solve(
        y, P, x0=x0, engine=engine, show=show, **kwargs_solver
    )
    return xinv, istop, itn, r1norm, r2norm
