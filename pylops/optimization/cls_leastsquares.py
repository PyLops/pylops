__all__ = [
    "NormalEquationsInversion",
    "RegularizedOperator",
    "RegularizedInversion",
    "PreconditionedInversion",
]

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import lsqr

from pylops.basicoperators import Diagonal, VStack
from pylops.optimization.basesolver import Solver
from pylops.optimization.basic import cg, cgls
from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _check_regularization_dims(
    Regs: Sequence["LinearOperator"],
    dataregs: Optional[Sequence[NDArray]] = None,
    epsRs: Optional[Sequence[float]] = None,
) -> None:
    """check Regs, dataregs, and epsRs have same dimensions"""
    nRegs = len(Regs)
    ndataregs = nRegs if dataregs is None else len(dataregs)
    nepsRs = nRegs if epsRs is None else len(epsRs)
    if not nRegs == ndataregs == nepsRs:
        raise ValueError("Regs, dataregs, and epsRs must have the same size")


class NormalEquationsInversion(Solver):
    r"""Inversion of normal equations.

    Solve the regularized normal equations for a system of equations
    given the operator ``Op``, a data weighting operator ``Weight`` and
    optionally a list of regularization terms ``Regs`` and/or ``NRegs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`.

    See Also
    --------
    RegularizedInversion: Regularized inversion
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    Solve the following normal equations for a system of regularized equations
    given the operator :math:`\mathbf{Op}`, a data weighting operator
    :math:`\mathbf{W}`, a list of regularization terms (:math:`\mathbf{R}_i`
    and/or :math:`\mathbf{N}_i`), the data :math:`\mathbf{y}` and
    regularization data :math:`\mathbf{y}_{\mathbf{R}_i}`, and the damping factors
    :math:`\epsilon_I`, :math:`\epsilon_{\mathbf{R}_i}` and :math:`\epsilon_{\mathbf{N}_i}`:

    .. math::
        ( \mathbf{Op}^T \mathbf{W} \mathbf{Op} +
        \sum_i \epsilon_{\mathbf{R}_i}^2 \mathbf{R}_i^T \mathbf{R}_i +
        \sum_i \epsilon_{\mathbf{N}_i}^2 \mathbf{N}_i +
        \epsilon_I^2 \mathbf{I} )  \mathbf{x}
        = \mathbf{Op}^T \mathbf{W} \mathbf{y} +  \sum_i \epsilon_{\mathbf{R}_i}^2
        \mathbf{R}_i^T \mathbf{y}_{\mathbf{R}_i}

    Note that the data term of the regularizations :math:`\mathbf{N}_i` is
    implicitly assumed to be zero.

    """

    def _print_setup(self) -> None:
        self._print_solver(nbar=55)
        strreg = f"Regs={self.Regs}"
        streps = f"\nepsRs={self.epsRs}     epsI={self.epsI}"
        print(strreg + streps)
        print("-" * 55)

    def _print_finalize(self) -> None:
        print(f"\nTotal time (s) = {self.telapsed:.2f}")
        print("-" * 55 + "\n")

    def setup(
        self,
        y: NDArray,
        Regs: Sequence["LinearOperator"],
        Weight: Optional["LinearOperator"] = None,
        dataregs: Optional[Sequence[NDArray]] = None,
        epsI: float = 0,
        epsRs: Optional[Sequence[float]] = None,
        NRegs: Optional[Sequence["LinearOperator"]] = None,
        epsNRs: Optional[Sequence[float]] = None,
        show: bool = False,
    ) -> None:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        Regs : :obj:`list`
            Regularization operators (``None`` to avoid adding regularization)
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
        show : :obj:`bool`, optional
            Display setup log

        """
        self.y = y
        self.Regs = Regs
        self.epsI = epsI
        self.epsRs = epsRs
        self.dataregs = dataregs
        self.ncp = get_array_module(y)

        # check consistency in regularization terms
        if Regs is not None:
            _check_regularization_dims(Regs, dataregs, epsRs)

        # store adjoint
        self.OpH = self.Op.H

        # create dataregs and epsRs if not provided
        if dataregs is None and Regs is not None:
            self.dataregs = [
                self.ncp.zeros(int(Reg.shape[0]), dtype=Reg.dtype) for Reg in Regs
            ]
        if epsRs is None and Regs is not None:
            self.epsRs = [1] * len(Regs)

        # normal equations
        if Weight is not None:
            self.y_normal = self.Op.rmatvec(Weight.matvec(y))
        else:
            self.y_normal = self.Op.rmatvec(y)
        if Weight is not None:
            self.Op_normal = self.OpH @ Weight @ self.Op
        else:
            self.Op_normal = self.OpH @ self.Op

        # add regularization terms
        if epsI > 0:
            self.Op_normal += epsI**2 * Diagonal(
                self.ncp.ones(self.Op.dims, dtype=self.Op.dtype),
                dtype=self.Op.dtype,
            )

        if (
            self.epsRs is not None
            and self.Regs is not None
            and self.dataregs is not None
        ):
            for epsR, Reg, datareg in zip(self.epsRs, self.Regs, self.dataregs):
                self.y_normal += epsR**2 * Reg.rmatvec(datareg)
                self.Op_normal += epsR**2 * Reg.H @ Reg

        if epsNRs is not None and NRegs is not None:
            for epsNR, NReg in zip(epsNRs, NRegs):
                self.Op_normal += epsNR**2 * NReg

        # print setup
        if show:
            self._print_setup()

    def step(self) -> None:
        raise NotImplementedError(
            "NormalEquationsInversion uses as default the"
            " scipy.sparse.linalg.cg solver, therefore the "
            "step method is not implemented. Use directly run or solve."
        )

    def run(
        self,
        x: NDArray,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of the solver.
            If ``None``, x is assumed to be a zero vector
        engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
        show : :obj:`bool`, optional
            Display iterations log
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.cg` and
            :py:func:`pylops.optimization.solver.cg` are used as default for numpy
            and cupy `data`, respectively)

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

        """
        if x is not None:
            self.y_normal = self.y_normal - self.Op_normal.matvec(x)
        if engine == "scipy" and self.ncp == np:
            if "atol" not in kwargs_solver:
                kwargs_solver["atol"] = "legacy"
            xinv, istop = sp_cg(self.Op_normal, self.y_normal, **kwargs_solver)
        elif engine == "pylops" or self.ncp != np:
            if show:
                kwargs_solver["show"] = True
            xinv = cg(
                self.Op_normal,
                self.y_normal,
                self.ncp.zeros(self.Op_normal.shape[1], dtype=self.Op_normal.dtype),
                **kwargs_solver,
            )[0]
            istop = None
        else:
            raise NotImplementedError("Engine must be scipy or pylops")
        if x is not None:
            xinv = x + xinv
        return xinv, istop

    def solve(
        self,
        y: NDArray,
        Regs: Sequence["LinearOperator"],
        x0: Optional[NDArray] = None,
        Weight: Optional["LinearOperator"] = None,
        dataregs: Optional[Sequence[NDArray]] = None,
        epsI: float = 0,
        epsRs: Optional[Sequence[float]] = None,
        NRegs: Optional[Sequence["LinearOperator"]] = None,
        epsNRs: Optional[Sequence[float]] = None,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        Regs : :obj:`list`
            Regularization operators (``None`` to avoid adding regularization)
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
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
            Display setup log

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[N \times 1]`
        istop : :obj:`int`
        Convergence information (only when using :py:func:`scipy.sparse.linalg.cg`):

        ``0``: successful exit

        ``>0``: convergence to tolerance not achieved, number of iterations

        ``<0``: illegal input or breakdown

        """
        self.setup(
            y=y,
            Regs=Regs,
            Weight=Weight,
            dataregs=dataregs,
            epsI=epsI,
            epsRs=epsRs,
            NRegs=NRegs,
            epsNRs=epsNRs,
            show=show,
        )
        x, istop = self.run(x0, engine=engine, show=show, **kwargs_solver)
        self.finalize(show)
        return x, istop


def RegularizedOperator(
    Op: "LinearOperator",
    Regs: Sequence["LinearOperator"],
    epsRs: Sequence[float] = (1,),
) -> "LinearOperator":
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
            \epsilon_{\mathbf{R}_1} \mathbf{R}_1 \\
            ...   \\
            \epsilon_{R_N} \mathbf{R}_N
        \end{bmatrix}

    """
    OpReg = VStack(
        [Op] + [epsR * Reg for epsR, Reg in zip(epsRs, Regs)], dtype=Op.dtype
    )
    return OpReg


class RegularizedInversion(Solver):
    r"""Regularized inversion.

    Solve a system of regularized equations given the operator ``Op``,
    a data weighting operator ``Weight``, and a list of regularization
    terms ``Regs``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`.

    See Also
    --------
    RegularizedOperator: Regularized operator
    NormalEquationsInversion: Normal equations inversion
    PreconditionedInversion: Preconditioned inversion

    Notes
    -----
    Solve the following system of regularized equations given the operator
    :math:`\mathbf{Op}`, a data weighting operator :math:`\mathbf{W}^{1/2}`,
    a list of regularization terms :math:`\mathbf{R}_i`,
    the data :math:`\mathbf{y}` and
    regularization data :math:`\mathbf{y}_{\mathbf{R}_i}`, and the damping factors
    :math:`\epsilon_\mathbf{I}`: and :math:`\epsilon_{\mathbf{R}_i}`:

    .. math::
        \begin{bmatrix}
            \mathbf{W}^{1/2} \mathbf{Op}    \\
            \epsilon_{\mathbf{R}_1} \mathbf{R}_1 \\
            \vdots   \\
            \epsilon_{\mathbf{R}_N} \mathbf{R}_N
        \end{bmatrix} \mathbf{x} =
        \begin{bmatrix}
            \mathbf{W}^{1/2} \mathbf{y}    \\
            \epsilon_{\mathbf{R}_1} \mathbf{y}_{\mathbf{R}_1} \\
            \vdots   \\
            \epsilon_{\mathbf{R}_N} \mathbf{y}_{\mathbf{R}_N} \\
        \end{bmatrix}

    where the ``Weight`` provided here is equivalent to the
    square-root of the weight in
    :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`. Note
    that this system is solved using the :py:func:`scipy.sparse.linalg.lsqr`
    and an initial guess ``x0`` can be provided to this solver, despite the
    original solver does not allow so.

    """

    def _print_setup(self) -> None:
        self._print_solver(nbar=65)
        strreg = f"Regs={self.Regs}"
        streps = f"\nepsRs={self.epsRs}"
        print(strreg + streps)
        print("-" * 65)

    def _print_finalize(self) -> None:
        print(f"\nTotal time (s) = {self.telapsed:.2f}")
        print("-" * 65 + "\n")

    def setup(
        self,
        y: NDArray,
        Regs: Sequence["LinearOperator"],
        Weight: Optional["LinearOperator"] = None,
        dataregs: Optional[Sequence[NDArray]] = None,
        epsRs: Optional[Sequence[float]] = None,
        show: bool = False,
    ) -> None:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        Regs : :obj:`list`
            Regularization operators (``None`` to avoid adding regularization)
        Weight : :obj:`pylops.LinearOperator`, optional
            Weight operator
        dataregs : :obj:`list`, optional
            Regularization data (must have the same number of elements
            as ``Regs``)
        epsRs : :obj:`list`, optional
             Regularization dampings (must have the same number of elements
             as ``Regs``)
        show : :obj:`bool`, optional
            Display setup log

        """
        self.y = y
        self.Regs = Regs
        self.epsRs = epsRs
        self.dataregs = dataregs
        self.ncp = get_array_module(y)

        # check consistency in regularization terms
        if Regs is not None:
            _check_regularization_dims(Regs, dataregs, epsRs)

        # create regularization data
        if dataregs is None and Regs is not None:
            self.dataregs = [
                self.ncp.zeros(int(Reg.shape[0]), dtype=Reg.dtype) for Reg in Regs
            ]

        if self.epsRs is None and Regs is not None:
            self.epsRs = [1] * len(Regs)

        # create regularization operators
        self.RegOp: LinearOperator
        if Weight is not None:
            if Regs is None:
                self.RegOp = Weight @ self.Op
            else:
                self.RegOp = RegularizedOperator(
                    Weight @ self.Op, Regs, epsRs=self.epsRs
                )
        else:
            if Regs is None:
                self.RegOp = self.Op
            else:
                self.RegOp = RegularizedOperator(self.Op, Regs, epsRs=self.epsRs)

        # augumented data
        if Weight is not None:
            self.datatot: NDArray = Weight.matvec(self.y.copy())
        else:
            self.datatot = self.y.copy()

        # augumented operator
        if self.epsRs is not None and self.dataregs is not None:
            for epsR, datareg in zip(self.epsRs, self.dataregs):
                self.datatot = np.hstack((self.datatot, epsR * datareg))

        # print setup
        if show:
            self._print_setup()

    def step(self) -> None:
        raise NotImplementedError(
            "RegularizedInversion uses as default the"
            " scipy.sparse.linalg.lsqr solver, therefore the "
            "step method is not implemented. Use directly run or solve."
        )

    def run(
        self,
        x: NDArray,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int, int, float, float]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of the solver.
            If ``None``, x is assumed to be a zero vector
        engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
        show : :obj:`bool`, optional
            Display iterations log
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

        """
        if x is not None:
            self.datatot = self.datatot - self.RegOp.matvec(x)
        if engine == "scipy" and self.ncp == np:
            if show:
                kwargs_solver["show"] = 1
            xinv, istop, itn, r1norm, r2norm = lsqr(
                self.RegOp, self.datatot, **kwargs_solver
            )[0:5]
        elif engine == "pylops" or self.ncp != np:
            if show:
                kwargs_solver["show"] = True
            xinv, istop, itn, r1norm, r2norm = cgls(
                self.RegOp,
                self.datatot,
                self.ncp.zeros(self.RegOp.shape[1], dtype=self.RegOp.dtype),
                **kwargs_solver,
            )[0:5]
        else:
            raise NotImplementedError("Engine must be scipy or pylops")
        if x is not None:
            xinv = x + xinv
        return xinv, istop, itn, r1norm, r2norm

    def solve(
        self,
        y: NDArray,
        Regs: Sequence["LinearOperator"],
        x0: Optional[NDArray] = None,
        Weight: Optional["LinearOperator"] = None,
        dataregs: Optional[Sequence[NDArray]] = None,
        epsRs: Optional[Sequence[float]] = None,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int, int, float, float]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        Regs : :obj:`list`
            Regularization operators (``None`` to avoid adding regularization)
        x0 : :obj:`numpy.ndarray`, optional
            Initial guess
        Weight : :obj:`pylops.LinearOperator`, optional
            Weight operator
        dataregs : :obj:`list`, optional
            Regularization data (must have the same number of elements
            as ``Regs``)
        epsRs : :obj:`list`, optional
             Regularization dampings (must have the same number of elements
             as ``Regs``)
        engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
        show : :obj:`bool`, optional
            Display log
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

        """
        self.setup(
            y=y, Regs=Regs, Weight=Weight, dataregs=dataregs, epsRs=epsRs, show=show
        )
        x, istop, itn, r1norm, r2norm = self.run(
            x0, engine=engine, show=show, **kwargs_solver
        )
        self.finalize(show)
        return x, istop, itn, r1norm, r2norm


class PreconditionedInversion(Solver):
    r"""Preconditioned inversion.

    Solve a system of preconditioned equations given the operator
    ``Op`` and a preconditioner ``P``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`.

    See Also
    --------
    RegularizedInversion: Regularized inversion
    NormalEquationsInversion: Normal equations inversion

    Notes
    -----
    Solve the following system of preconditioned equations given the operator
    :math:`\mathbf{Op}`, a preconditioner :math:`\mathbf{P}`,
    the data :math:`\mathbf{y}`

    .. math::
        \mathbf{y} = \mathbf{Op}\,\mathbf{P} \mathbf{p}

    where :math:`\mathbf{p}` is the solution in the preconditioned space
    and :math:`\mathbf{x} = \mathbf{P}\mathbf{p}` is the solution in the
    original space.

    """

    def _print_setup(self) -> None:
        self._print_solver(nbar=65)
        strprec = f"Prec={self.P}"
        print(strprec)
        print("-" * 65)

    def _print_finalize(self) -> None:
        print(f"\nTotal time (s) = {self.telapsed:.2f}")
        print("-" * 65 + "\n")

    def setup(
        self,
        y: NDArray,
        P: "LinearOperator",
        show: bool = False,
    ) -> None:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        P : :obj:`pylops.LinearOperator`
            Preconditioner
        show : :obj:`bool`, optional
            Display setup log

        """
        self.y = y
        self.P = P
        self.ncp = get_array_module(y)

        # preconditioned operator
        self.POp = self.Op @ P

        # print setup
        if show:
            self._print_setup()

    def step(self) -> None:
        raise NotImplementedError(
            "PreconditionedInversion uses as default the"
            " scipy.sparse.linalg.lsqr solver, therefore the "
            "step method is not implemented. Use directly run or solve."
        )

    def run(
        self,
        x: NDArray,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int, int, float, float]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of the solver.
            If ``None``, x is assumed to be a zero vector
        engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
        show : :obj:`bool`, optional
            Display iterations log
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

        """
        if x is not None:
            self.y = self.y - self.Op.matvec(x)
        if engine == "scipy" and self.ncp == np:
            if show:
                kwargs_solver["show"] = 1
            pinv, istop, itn, r1norm, r2norm = lsqr(
                self.POp,
                self.y,
                **kwargs_solver,
            )[0:5]
        elif engine == "pylops" or self.ncp != np:
            if show:
                kwargs_solver["show"] = True
            pinv, istop, itn, r1norm, r2norm = cgls(
                self.POp,
                self.y,
                self.ncp.zeros(self.POp.shape[1], dtype=self.POp.dtype),
                **kwargs_solver,
            )[0:5]
            # force it 1d as we decorate this method with disable_ndarray_multiplication
            pinv = pinv.ravel()
        else:
            raise NotImplementedError("Engine must be scipy or pylops")
        xinv = self.P.matvec(pinv)
        if x is not None:
            xinv = x + xinv
        return xinv, istop, itn, r1norm, r2norm

    def solve(
        self,
        y: NDArray,
        P: "LinearOperator",
        x0: Optional[NDArray] = None,
        engine: str = "scipy",
        show: bool = False,
        **kwargs_solver,
    ) -> Tuple[NDArray, int, int, float, float]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        P : :obj:`pylops.LinearOperator`
            Preconditioner
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
        engine : :obj:`str`, optional
            Solver to use (``scipy`` or ``pylops``)
        show : :obj:`bool`, optional
            Display log
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.lsqr` and
            :py:func:`pylops.optimization.solver.cgls` are used for engine ``scipy``
            and ``pylops``, respectively)

        Returns
        -------
        x : :obj:`numpy.ndarray`
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

        """
        self.setup(y=y, P=P, show=show)
        x, istop, itn, r1norm, r2norm = self.run(
            x0, engine=engine, show=show, **kwargs_solver
        )
        self.finalize(show)
        return x, istop, itn, r1norm, r2norm
