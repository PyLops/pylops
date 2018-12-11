import numpy as np
from scipy.sparse.linalg import cg, lsqr

from pylops.basicoperators import MatrixMult
from pylops.basicoperators import VStack


def NormalEquationsInversion(Op, Regs, data, dataregs=None, epsI=0,
                             epsRs=None, x0=None,
                             returninfo=False, **kwargs_cg):
    r"""Inversion of normal equations.

    Solve the regularized normal equations for a system of equations given the operator ``Op`` and a
    list of regularization terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    Regs : :obj:`list`
        Regularization operators
    data : :obj:`numpy.ndarray`
        Data
    dataregs : :obj:`list`, optional
        Regularization data
    espI : :obj:`float`
        Tikhonov damping, optional
    epsRs : :obj:`list`
         Regularization dampings (must have the same number of elements
         as ``Regs``)
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    returninfo : :obj:`bool`, optional
        Return info of CG solver
    **kwargs_cg
        Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.cg` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.
    istop : :obj:`int`
        Convergence information:

        ``0``: successful exit

        ``>0``: convergence to tolerance not achieved, number of iterations

        ``<0``: illegal input or breakdown

    Notes
    -----
    Solve the following normal equations for a system of regularized equations
    given the operator :math:`\mathbf{Op}`, a list of regularization terms
    :math:`\mathbf{R_i}`, the data :math:`\mathbf{d}` and regularization damping
    factors :math:`\epsilon_I` and :math:`\epsilon_{{R}_i}`:

    .. math::
        ( \mathbf{Op}^T\mathbf{Op} + \sum_i \epsilon_{{R}_i}^2
        \mathbf{R}_i^T \mathbf{R}_i + \epsilon_I^2 \mathbf{I} )  \mathbf{x}
        = \mathbf{Op}^T \mathbf{y} +  \sum_i \epsilon_{{R}_i}^2
        \mathbf{R}_i^T \mathbf{d}_{R_i}

    """
    if dataregs is None:
        dataregs = [np.zeros(Op.shape[1])]*len(Regs)

    if epsRs is None:
        epsRs = [1] * len(Regs)

    # Normal equations
    y_normal = Op.H * data
    if Regs is not None:
        for epsR, Reg, datareg in zip(epsRs, Regs, dataregs):
            y_normal += epsR ** 2 * Reg.H * datareg
    Op_normal = Op.H * Op
    if epsI > 0:
        Op_normal += epsI ** 2 * MatrixMult(np.eye(Op.shape[1]))
    if Regs is not None:
        for epsR, Reg in zip(epsRs, Regs):
            Op_normal += epsR ** 2 * Reg.H * Reg

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
    :math:`\mathbf{Op}`, by a set of regularization terms :math:`\mathbf{R_i}` and their
    damping factors and :math:`\epsilon_{{R}_i}`:

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


def RegularizedInversion(Op, Regs, data, dataregs=None, epsRs=None,
                         x0=None, returninfo=False, **kwargs_lsqr):
    r"""Regularized inversion.

    Solve a system of regularized equations given the operator ``Op``
    and a list of regularization terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    Regs : :obj:`list`
        Regularization operators
    data : :obj:`numpy.ndarray`
        Data
    dataregs : :obj:`list`, optional
        Regularization data
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

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2`, where
        :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +  \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`

    See Also
    --------
    RegularizedOperator: Regularized operator
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    Solve the following system of regularized equations given the operator
    :math:`\mathbf{Op}`, a list of regularization terms :math:`\mathbf{R_i}`,
    the data :math:`\mathbf{d}` and regularization damping factors
    :math:`\epsilon_I`: and :math:`\epsilon_{{R}_i}`:

    .. math::
        \begin{bmatrix}
            \mathbf{Op}    \\
            \epsilon_{R_1} \mathbf{R}_1 \\
            ...   \\
            \epsilon_{R_N} \mathbf{R}_N
        \end{bmatrix} \mathbf{x} =
        \begin{bmatrix}
            \mathbf{d}    \\
            \epsilon_{R_1} \mathbf{d}_{R_1} \\
            ...   \\
            \epsilon_{R_N} \mathbf{d}_{R_N} \\
        \end{bmatrix}

    """
    # regularized operator
    if dataregs is None:
        dataregs = [np.zeros(Op.shape[1])] * len(Regs)

    if epsRs is None:
        epsRs = [1] * len(Regs)

    # operator
    RegOp = RegularizedOperator(Op, Regs, epsRs=epsRs)

    # augumented data
    datatot = data.copy()
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


def PreconditionedInversion(Op, P, data, x0=None, returninfo=False, **kwargs_lsqr):
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
    xinv : :obj:`numpy.ndarray`
        Inverted model :math:`\mathbf{Op}`
    istop : :obj:`int`
        Gives the reason for termination

        ``1`` means :math:`\mathbf{x}` is an approximate solution to
        :math:`\mathbf{d} = \mathbf{Op}\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares problem
    itn : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2`, where :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +  \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`

    See Also
    --------
    RegularizedInversion: Regularized inversion

    Notes
    -----
    Solve the following system of preconditioned equations given the operator
    :math:`\mathbf{Op}`, a preconditioner :math:`\mathbf{P}`,
    the data :math:`\mathbf{d}`

    .. math::
        \mathbf{y} = \mathbf{Op} (\mathbf{P} \mathbf{p})

    where :math:`\mathbf{p}` is the solution in the preconditioned space
    and :math:`\mathbf{x} = \mathbf{P}\mathbf{p}` is the solution in the original space.

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
