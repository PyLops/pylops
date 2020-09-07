import numpy as np
from scipy.sparse.linalg import cg, lsqr

from pylops.basicoperators import Diagonal
from pylops.basicoperators import VStack


def NormalEquationsInversion(Op, Regs, data, Weight=None, dataregs=None,
                             epsI=0, epsRs=None, x0=None,
                             returninfo=False, NRegs=None, epsNRs=None,
                             **kwargs_cg):
    r"""Inversion of normal equations.

    Solve the regularized normal equations for a system of equations
    given the operator ``Op``, a data weighting operator ``Weight`` and
    optionally a list of regularization terms ``Regs`` and/or ``NRegs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    Regs : :obj:`list`
        Regularization operators (``None`` to avoid adding regularization)
    data : :obj:`numpy.ndarray`
        Data
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
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    returninfo : :obj:`bool`, optional
        Return info of CG solver
    NRegs : :obj:`list`
        Normal regularization operators (``None`` to avoid adding
        regularization). Such operators must apply the chain of the
        forward and the adjoint in one go. This can be convenient in
        cases where a faster implementation is available compared to applying
        the forward followed by the adjoint.
    epsNRs : :obj:`list`, optional
         Regularization dampings for normal operators (must have the same
         number of elements as ``NRegs``)
    **kwargs_cg
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.cg` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Convergence information:

        ``0``: successful exit

        ``>0``: convergence to tolerance not achieved, number of iterations

        ``<0``: illegal input or breakdown

    See Also
    --------
    RegularizedInversion: Regularized inversion
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    Solve the following normal equations for a system of regularized equations
    given the operator :math:`\mathbf{Op}`, a data weighting operator
    :math:`\mathbf{W}`, a list of regularization terms (:math:`\mathbf{R_i}`
    and/or :math:`\mathbf{N_i}`), the data :math:`\mathbf{d}` and
    regularization data :math:`\mathbf{d}_{R_i}`, and the damping factors
    :math:`\epsilon_I`, :math:`\epsilon_{{R}_i}` and :math:`\epsilon_{{N}_i}`:

    .. math::
        ( \mathbf{Op}^T \mathbf{W} \mathbf{Op} +
        \sum_i \epsilon_{{R}_i}^2 \mathbf{R}_i^T \mathbf{R}_i +
        \sum_i \epsilon_{{N}_i}^2 \mathbf{N}_i +
        \epsilon_I^2 \mathbf{I} )  \mathbf{x}
        = \mathbf{Op}^T \mathbf{W} \mathbf{d} +  \sum_i \epsilon_{{R}_i}^2
        \mathbf{R}_i^T \mathbf{d}_{R_i}

    Note that the data term of the regularizations :math:`\mathbf{N_i}` is
    implicitly assumed to be zero.

    """
    # store adjoint
    OpH = Op.H

    # create dataregs and epsRs if not provided
    if dataregs is None and Regs is not None:
        dataregs = [np.zeros(Reg.shape[0]) for Reg in Regs]

    if epsRs is None and Regs is not None:
        epsRs = [1] * len(Regs)

    # Normal equations
    if Weight is not None:
        y_normal = OpH * Weight * data
    else:
        y_normal = OpH * data
    if Weight is not None:
        Op_normal = OpH * Weight * Op
    else:
        Op_normal = OpH * Op

    # Add regularization terms
    if epsI > 0:
        Op_normal += epsI ** 2 * Diagonal(np.ones(Op.shape[1]))

    if Regs is not None:
        for epsR, Reg, datareg in zip(epsRs, Regs, dataregs):
            RegH = Reg.H
            y_normal += epsR ** 2 * RegH * datareg
            Op_normal += epsR ** 2 * RegH * Reg

    if NRegs is not None:
        for epsNR, NReg in zip(epsNRs, NRegs):
            Op_normal += epsNR ** 2 * NReg

    # CG solver
    if x0 is not None:
        y_normal = y_normal - Op_normal*x0
    xinv, istop = cg(Op_normal, y_normal, **kwargs_cg)
    if x0 is not None:
        xinv = x0 + xinv

    if returninfo:
        return xinv, istop
    else:
        return xinv


def RegularizedOperator(Op, Regs, epsRs=(1,)):
    r"""Regularized operator.

    Creates a regularized operator given the operator ``Op``
    and a list of regularization terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    Regs : :obj:`tuple` or :obj:`list`
        Regularization operators
    epsRs : :obj:`tuple` or :obj:`list`, optional
         Regularization dampings

    Returns
    -------
    OpReg : :obj:`pylops.LinearOperator`
        Regularized operator

    See Also
    --------
    RegularizedInversion: Regularized inversion

    Notes
    -----
    Create a regularized operator by augumenting the problem operator
    :math:`\mathbf{Op}`, by a set of regularization terms :math:`\mathbf{R_i}`
    and their damping factors and :math:`\epsilon_{{R}_i}`:

    .. math::
        \begin{bmatrix}
            \mathbf{Op}    \\
            \epsilon_{R_1} \mathbf{R}_1 \\
            ...   \\
            \epsilon_{R_N} \mathbf{R}_N
        \end{bmatrix}

    """
    OpReg = VStack([Op] + [epsR * Reg for epsR, Reg in zip(epsRs, Regs)],
                   dtype=Op.dtype)
    return OpReg


def RegularizedInversion(Op, Regs, data, Weight=None, dataregs=None,
                         epsRs=None, x0=None, returninfo=False, **kwargs_lsqr):
    r"""Regularized inversion.

    Solve a system of regularized equations given the operator ``Op``,
    a data weighting operator ``Weight``, and a list of regularization
    terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    Regs : :obj:`list`
        Regularization operators (``None`` to avoid adding regularization)
    data : :obj:`numpy.ndarray`
        Data
    Weight : :obj:`pylops.LinearOperator`, optional
        Weight operator
    dataregs : :obj:`list`, optional
        Regularization data (if ``None`` a zero data will be used for every
        regularization operator in ``Regs``)
    epsRs : :obj:`list`, optional
         Regularization dampings
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    returninfo : :obj:`bool`, optional
        Return info of LSQR solver
    **kwargs_lsqr
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model :math:`\mathbf{Op}`
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{d} = \mathbf{Op}\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where
        :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\mathbf{x}`
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
    Solve the following system of regularized equations given the operator
    :math:`\mathbf{Op}`, a data weighting operator :math:`\mathbf{W}^{1/2}`,
    a list of regularization terms :math:`\mathbf{R_i}`,
    the data :math:`\mathbf{d}` and regularization damping factors
    :math:`\epsilon_I`: and :math:`\epsilon_{{R}_i}`:

    .. math::
        \begin{bmatrix}
            \mathbf{W}^{1/2} \mathbf{Op}    \\
            \epsilon_{R_1} \mathbf{R}_1 \\
            ...   \\
            \epsilon_{R_N} \mathbf{R}_N
        \end{bmatrix} \mathbf{x} =
        \begin{bmatrix}
            \mathbf{W}^{1/2} \mathbf{d}    \\
            \epsilon_{R_1} \mathbf{d}_{R_1} \\
            ...   \\
            \epsilon_{R_N} \mathbf{d}_{R_N} \\
        \end{bmatrix}

    where the ``Weight`` provided here is equivalent to the
    square-root of the weight in
    :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`. Note
    that this system is solved using the :py:func:`scipy.sparse.linalg.lsqr`
    and an initial guess ``x0`` can be provided to this solver, despite the
    original solver does not allow so.

    """
    # create regularization data
    if dataregs is None and Regs is not None:
        dataregs = [np.zeros(Reg.shape[0]) for Reg in Regs]

    if epsRs is None and Regs is not None:
        epsRs = [1] * len(Regs)

    # create regularization operators
    if Weight is not None:
        if Regs is None:
            RegOp = Weight * Op
        else:
            RegOp = RegularizedOperator(Weight * Op, Regs, epsRs=epsRs)
    else:
        if Regs is None:
            RegOp = Op
        else:
            RegOp = RegularizedOperator(Op, Regs, epsRs=epsRs)

    # augumented data
    if Weight is not None:
        datatot = Weight*data.copy()
    else:
        datatot = data.copy()

    # augumented operator
    if Regs is not None:
        for epsR, datareg in zip(epsRs, dataregs):
            datatot = np.hstack((datatot, epsR*datareg))

    # LSQR solver
    if x0 is not None:
        datatot = datatot - RegOp * x0
    xinv, istop, itn, r1norm, r2norm = lsqr(RegOp, datatot, **kwargs_lsqr)[0:5]
    if x0 is not None:
        xinv = x0 + xinv

    if returninfo:
        return xinv, istop, itn, r1norm, r2norm
    else:
        return xinv


def PreconditionedInversion(Op, P, data, x0=None, returninfo=False,
                            **kwargs_lsqr):
    r"""Preconditioned inversion.

    Solve a system of preconditioned equations given the operator
    ``Op`` and a preconditioner ``P``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    P : :obj:`pylops.LinearOperator`
        Preconditioner
    data : :obj:`numpy.ndarray`
        Data
    x0 : :obj:`numpy.ndarray`
        Initial guess
    returninfo : :obj:`bool`
        Return info of LSQR solver
    **kwargs_lsqr
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{d} = \mathbf{Op}\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2^2`, where :math:`\mathbf{r} = \mathbf{d} -
        \mathbf{Op}\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +  \epsilon^2
        \mathbf{x}^T\mathbf{x}}`. Equal to ``r1norm`` if :math:`\epsilon=0`

    See Also
    --------
    RegularizedInversion: Regularized inversion
    NormalEquationsInversion: Normal equations inversion

    Notes
    -----
    Solve the following system of preconditioned equations given the operator
    :math:`\mathbf{Op}`, a preconditioner :math:`\mathbf{P}`,
    the data :math:`\mathbf{d}`

    .. math::
        \mathbf{d} = \mathbf{Op} (\mathbf{P} \mathbf{p})

    where :math:`\mathbf{p}` is the solution in the preconditioned space
    and :math:`\mathbf{x} = \mathbf{P}\mathbf{p}` is the solution in the
    original space.

    """
    # Preconditioned operator
    POp = Op*P
    # LSQR solver
    if x0 is not None:
        data = data - Op * x0

    pinv, istop, itn, r1norm, r2norm = lsqr(POp, data, **kwargs_lsqr)[0:5]
    xinv = P*pinv
    if x0 is not None:
        xinv = xinv + x0

    if returninfo:
        return xinv, istop, itn, r1norm, r2norm
    else:
        return xinv
