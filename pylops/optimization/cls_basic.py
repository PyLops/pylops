__all__ = [
    "CG",
    "CGLS",
    "LSQR",
]

import time
from typing import List, Optional, Tuple

import numpy as np

from pylops.optimization.basesolver import Solver
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray


class CG(Solver):
    r"""Conjugate gradient

    Solve a square system of equations given an operator ``Op`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times N]`

    Notes
    -----
    Solve the :math:`\mathbf{y} = \mathbf{Op}\,\mathbf{x}` problem using conjugate gradient
    iterations [1]_.

    .. [1] Hestenes, M R., Stiefel, E., “Methods of Conjugate Gradients for Solving
       Linear Systems”, Journal of Research of the National Bureau of Standards.
       vol. 49. 1952.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=55)

        if self.niter is not None:
            strpar = f"tol = {self.tol:10e}\tniter = {self.niter}"
        else:
            strpar = f"tol = {self.tol:10e}"
        print(strpar)
        print("-" * 55 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              r2norm"
        else:
            head1 = "    Itn              x[0]                  r2norm"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        strx = f"{x[0]:1.2e}        " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = f"{self.iiter:6g}        " + strx + f"{self.cost[self.iiter]:11.4e}"
        print(msg)

    def setup(
        self,
        y: NDArray,
        x0: Optional[NDArray] = None,
        niter: Optional[int] = None,
        tol: float = 1e-4,
        show: bool = False,
    ) -> NDArray:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[N \times 1]`. If ``None``, initialize
            internally as zero vector
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`np.ndarray`
            Initial guess of size :math:`[N \times 1]`

        """
        self.y = y
        self.niter = niter
        self.tol = tol
        self.ncp = get_array_module(y)

        # initialize solver
        if x0 is None:
            x = self.ncp.zeros(self.Op.shape[1], dtype=self.y.dtype)
            self.r = self.y.copy()
        else:
            x = x0.copy()
            self.r = self.y - self.Op.matvec(x)
        self.c = self.r.copy()
        self.kold = self.ncp.abs(self.r.dot(self.r.conj()))

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost.append(np.sqrt(self.kold))
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x))
        return x

    def step(self, x: NDArray, show: bool = False) -> NDArray:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector

        """
        Opc = self.Op.matvec(self.c)
        cOpc = self.ncp.abs(self.c.dot(Opc.conj()))
        a = self.kold / cOpc
        x += a * self.c
        self.r -= a * Opc
        k = self.ncp.abs(self.r.dot(self.r.conj()))
        b = k / self.kold
        self.c = self.r + b * self.c
        self.kold = k
        self.iiter += 1
        self.cost.append(np.sqrt(self.kold))
        if show:
            self._print_step(x)
        return x

    def run(
        self,
        x: NDArray,
        niter: Optional[int] = None,
        show: bool = False,
        itershow: List[int] = [10, 10, 10],
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of CG
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        niter = self.niter if niter is None else niter
        while self.iiter < niter and self.kold > self.tol:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x = self.step(x, showstep)
            self.callback(x)
        return x

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        self.cost = np.array(self.cost)
        if show:
            self._print_finalize(nbar=55)

    def solve(
        self,
        y: NDArray,
        x0: Optional[NDArray] = None,
        niter: int = 10,
        tol: float = 1e-4,
        show: bool = False,
        itershow: List[int] = [10, 10, 10],
    ) -> Tuple[NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[N \times 1]`. If ``None``, initialize
            internally as zero vector
        niter : :obj:`int`, optional
            Number of iterations
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[N \times 1]`
        iit : :obj:`int`
            Number of executed iterations
        cost : :obj:`numpy.ndarray`
            History of the L2 norm of the residual

        """
        x = self.setup(y=y, x0=x0, niter=niter, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost


class CGLS(Solver):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``Op`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`

    Notes
    -----
    Minimize the following functional using conjugate gradient iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Op}\,\mathbf{x} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        if self.niter is not None:
            strpar = (
                f"damp = {self.damp:10e}\ttol = {self.tol:10e}\tniter = {self.niter}"
            )
        else:
            strpar = f"damp = {self.damp:10e}\ttol = {self.tol:10e}\t"
        print(strpar)
        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn          x[0]              r1norm         r2norm"
        else:
            head1 = "    Itn             x[0]             r1norm         r2norm"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = (
            f"{self.iiter:6g}       "
            + strx
            + f"{self.cost[self.iiter]:11.4e}    {self.cost1[self.iiter]:11.4e}"
        )
        print(msg)

    def setup(
        self,
        y: NDArray,
        x0: Optional[NDArray] = None,
        niter: Optional[int] = None,
        damp: float = 0.0,
        tol: float = 1e-4,
        show: bool = False,
    ) -> NDArray:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess  of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
        damp : :obj:`float`, optional
            Damping coefficient
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`np.ndarray`
            Initial guess of size :math:`[N \times 1]`

        """
        self.y = y
        self.damp = damp**2
        self.tol = tol
        self.niter = niter
        self.ncp = get_array_module(y)

        # initialize solver
        if x0 is None:
            x = self.ncp.zeros(self.Op.shape[1], dtype=y.dtype)
            self.s = self.y.copy()
            r = self.Op.rmatvec(self.s)
        else:
            x = x0.copy()
            self.s = self.y - self.Op.matvec(x)
            r = self.Op.rmatvec(self.s) - damp * x
        self.c = r.copy()
        self.q = self.Op.matvec(self.c)
        self.kold = self.ncp.abs(r.dot(r.conj()))

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost1 = []
        self.cost.append(float(self.ncp.linalg.norm(self.s)))
        self.cost1.append(
            float(
                self.ncp.sqrt(self.cost[0] ** 2 + damp * self.ncp.abs(x.dot(x.conj())))
            )
        )
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x))
        return x

    def step(self, x: NDArray, show: bool = False) -> NDArray:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        """
        a = self.kold / (
            self.q.dot(self.q.conj()) + self.damp * self.c.dot(self.c.conj())
        )
        x = x + a * self.c
        self.s = self.s - a * self.q
        r = self.Op.rmatvec(self.s) - self.damp * x
        k = self.ncp.abs(r.dot(r.conj()))
        b = k / self.kold
        self.c = r + b * self.c
        self.q = self.Op.matvec(self.c)
        self.kold = k
        self.iiter += 1
        self.cost.append(float(self.ncp.linalg.norm(self.s)))
        self.cost1.append(
            self.ncp.sqrt(
                float(
                    self.cost[self.iiter] ** 2
                    + self.damp * self.ncp.abs(x.dot(x.conj()))
                )
            )
        )
        if show:
            self._print_step(x)
        return x

    def run(
        self,
        x: NDArray,
        niter: Optional[int] = None,
        show: bool = False,
        itershow: List[int] = [10, 10, 10],
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of CGLS
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display iterations log
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        niter = self.niter if niter is None else niter
        while self.iiter < niter and self.kold > self.tol:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x = self.step(x, showstep)
            self.callback(x)
        return x

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        # reason for termination
        self.istop = 1 if self.kold < self.tol else 2
        self.r1norm = self.kold
        self.r2norm = self.cost1[self.iiter]
        if show:
            self._print_finalize(nbar=65)
        self.cost = np.array(self.cost)

    def solve(
        self,
        y: NDArray,
        x0: Optional[NDArray] = None,
        niter: int = 10,
        damp: float = 0.0,
        tol: float = 1e-4,
        show: bool = False,
        itershow: List[int] = [10, 10, 10],
    ) -> Tuple[NDArray, int, int, float, float, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`
            Initial guess  of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
        niter : :obj:`int`, optional
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
        damp : :obj:`float`, optional
            Damping coefficient
        tol : :obj:`float`, optional
            Tolerance on residual norm
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

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

        """
        x = self.setup(y=y, x0=x0, niter=niter, damp=damp, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.istop, self.iiter, self.r1norm, self.r2norm, self.cost


class LSQR(Solver):
    r"""LSQR

    Solve an overdetermined system of equations given an operator ``Op`` and
    data ``y`` using LSQR iterations.

    .. math::
      \DeclareMathOperator{\cond}{cond}

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times M]`

    Notes
    -----
    Minimize the following functional using LSQR iterations [1]_:

    .. math::
        J = || \mathbf{y} -  \mathbf{Op}\,\mathbf{x} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    .. [1] Paige, C. C., and Saunders, M. A. "LSQR: An algorithm for sparse
        linear equations and sparse least squares", ACM TOMS, vol. 8, pp. 43-71,
        1982.

    """

    def __init__(self, Op):
        super().__init__(Op)
        self.msg = (
            "The exact solution is x = 0                               ",
            "Op x - b is small enough, given atol, btol                 ",
            "The least-squares solution is good enough, given atol     ",
            "The estimate of cond(Opbar) has exceeded conlim            ",
            "Op x - b is small enough for this machine                  ",
            "The least-squares solution is good enough for this machine",
            "Cond(Opbar) seems to be too large for this machine         ",
            "The iteration limit has been reached                      ",
        )

    def _print_setup(self, x: NDArray, xcomplex: bool = False) -> None:
        self._print_solver(nbar=90)
        print(f"damp = {self.damp:20.14e}     calc_var = {self.calc_var:6g}")
        print(f"atol = {self.atol:8.2e}                 conlim = {self.conlim:8.2e}")
        if self.niter is not None:
            strpar = f"btol = {self.btol:8.2e}                 niter = {self.niter:8g}"
        else:
            strpar = f"btol = {self.btol:8.2e}"
        print(strpar)
        print("-" * 90)
        head2 = " Compatible   LS     Norm A   Cond A"
        if not xcomplex:
            head1 = "   Itn     x[0]      r1norm     r2norm  "
        else:
            head1 = "   Itn        x[0]              r1norm    r2norm  "
        print(head1 + head2)
        test1 = 1
        test2 = self.alfa / self.beta
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}"
        str1 = f"{0:6g} " + strx
        str2 = f" {self.r1norm:10.3e} {self.r2norm:10.3e}"
        str3 = f"  {test1:8.1e} {test2:8.1e}"
        print(str1 + str2 + str3)

    def _print_step(self, x: NDArray) -> None:
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}"
        str1 = f"{self.iiter:6g} " + strx
        str2 = f" {self.r1norm:10.3e} {self.r2norm:10.3e}"
        str3 = f"  {self.test1:8.1e} {self.test2:8.1e}"
        str4 = f" {self.anorm:8.1e} {self.acond:8.1e}"
        print(str1 + str2 + str3 + str4)

    def _print_finalize(self) -> None:
        print(" ")
        print(f"LSQR finished, {self.msg[self.istop]}")
        print(" ")
        str1 = f"istop ={self.istop:8g}   r1norm ={self.r1norm:8.1e}"
        str2 = f"anorm ={self.anorm:8.1e}   arnorm ={self.arnorm:8.1e}"
        str3 = f"itn   ={self.iiter:8g}   r2norm ={self.r2norm:8.1e}"
        str4 = f"acond ={self.acond:8.1e}   xnorm  ={self.xnorm:8.1e}"
        str5 = f"Total time (s) = {self.telapsed:.2f}"
        print(str1 + "   " + str2)
        print(str3 + "   " + str4)
        print(str5)
        print("-" * 90 + "\n")

    def setup(
        self,
        y: NDArray,
        x0: Optional[NDArray] = None,
        damp: float = 0.0,
        atol: float = 1e-08,
        btol: float = 1e-08,
        conlim: float = 100000000.0,
        niter: int = 10,
        calc_var: bool = True,
        show: bool = False,
    ) -> NDArray:
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
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
            Display setup log

        Returns
        -------
        x : :obj:`np.ndarray`
            Initial guess of size :math:`[N \times 1]`

        """
        self.y = y
        self.damp = damp
        self.atol = atol
        self.btol = btol
        self.conlim = conlim
        self.niter = niter
        self.calc_var = calc_var
        self.ncp = get_array_module(y)

        m, n = self.Op.shape

        # initialize solver
        self.var = None
        if self.calc_var:
            self.var = self.ncp.zeros(n)

        self.iiter = 0
        self.istop = 0
        self.ctol = 0
        if conlim > 0:
            self.ctol = 1.0 / conlim
        self.anorm = 0
        self.acond = 0
        self.dampsq = damp**2
        self.ddnorm = 0
        self.res2 = 0
        self.xnorm = 0
        self.xxnorm = 0
        self.z = 0
        self.cs2 = -1
        self.sn2 = 0

        # initialize x0 and set up the first vectors u and v for the
        # bidiagonalization. These satisfy beta*u=b-Op(x0), alfa*v=Op'u
        if x0 is None:
            x = self.ncp.zeros(self.Op.shape[1], dtype=y.dtype)
            self.u = y.copy()
        else:
            x = x0.copy()
            self.u = self.y - self.Op.matvec(x0)
        self.alfa = 0.0
        self.beta = self.ncp.linalg.norm(self.u)
        if self.beta > 0.0:
            self.u = self.u / self.beta
            self.v = self.Op.rmatvec(self.u)
            self.alfa = self.ncp.linalg.norm(self.v)
            if self.alfa > 0:
                self.v = self.v / self.alfa
        else:
            self.v = x.copy()
            self.alfa = 0
        self.w = self.v.copy()

        # check if solution is already found
        self.arnorm = self.alfa * self.beta

        # finalize setup
        self.arnorm0 = self.arnorm
        self.rhobar = self.alfa
        self.phibar = self.beta
        self.bnorm = self.beta
        self.rnorm = self.beta
        self.r1norm = self.rnorm
        self.r2norm = self.rnorm

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost.append(float(self.rnorm))

        # print setup
        if show:
            self._print_setup(x, np.iscomplexobj(x))
        if self.arnorm == 0:
            print(" ")
            print("LSQR finished")
            print(self.msg[self.istop])
        return x

    def step(self, x: NDArray, show: bool = False) -> NDArray:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        # perform the next step of the bidiagonalization to obtain the
        # next beta, u, alfa, v. These satisfy the relations
        # beta*u = Op*v - alfa*u,
        # alfa*v = Op'*u - beta*v'
        self.u = self.Op.matvec(self.v) - self.alfa * self.u
        self.beta = self.ncp.linalg.norm(self.u)
        if self.beta > 0:
            self.u = self.u / self.beta
            self.anorm = np.linalg.norm(
                [self.anorm, to_numpy(self.alfa), to_numpy(self.beta), self.damp]
            )
            self.v = self.Op.rmatvec(self.u) - self.beta * self.v
            self.alfa = self.ncp.linalg.norm(self.v)
            if self.alfa > 0:
                self.v = self.v / self.alfa

        # use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        self.rhobar1 = np.linalg.norm([to_numpy(self.rhobar), self.damp])
        self.cs1 = self.rhobar / self.rhobar1
        self.sn1 = self.damp / self.rhobar1
        self.psi = self.sn1 * self.phibar
        self.phibar = self.cs1 * self.phibar

        # use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        self.rho = np.linalg.norm([self.rhobar1, to_numpy(self.beta)])
        self.cs = self.rhobar1 / self.rho
        self.sn = self.beta / self.rho
        self.theta = self.sn * self.alfa
        self.rhobar = -self.cs * self.alfa
        self.phi = self.cs * self.phibar
        self.phibar = self.sn * self.phibar
        self.tau = self.sn * self.phi

        # update x and w.
        self.t1 = self.phi / self.rho
        self.t2 = -self.theta / self.rho
        self.dk = self.w / self.rho
        x = x + self.t1 * self.w
        self.w = self.v + self.t2 * self.w
        self.ddnorm = self.ddnorm + self.ncp.linalg.norm(self.dk) ** 2
        if self.calc_var:
            self.var = self.var + self.ncp.dot(self.dk, self.dk)

        # use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        self.delta = self.sn2 * self.rho
        self.gambar = -self.cs2 * self.rho
        self.rhs = self.phi - self.delta * self.z
        self.zbar = self.rhs / self.gambar
        self.xnorm = self.ncp.sqrt(self.xxnorm + self.zbar**2)
        self.gamma = np.linalg.norm([self.gambar, to_numpy(self.theta)])
        self.cs2 = self.gambar / self.gamma
        self.sn2 = self.theta / self.gamma
        self.z = self.rhs / self.gamma
        self.xxnorm = self.xxnorm + self.z**2.0

        # test for convergence. First, estimate the condition of the matrix
        # Opbar, and the norms of rbar and Opbar'rbar
        self.acond = self.anorm * self.ncp.sqrt(self.ddnorm)
        self.res1 = self.phibar**2
        self.res2 = self.res2 + self.psi**2
        self.rnorm = self.ncp.sqrt(self.res1 + self.res2)
        self.arnorm = self.alfa * abs(self.tau)

        # distinguish between r1norm = ||b - Ax|| and
        # r2norm = sqrt(r1norm^2 + damp^2*||x||^2).
        # Estimate r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        self.r1sq = self.rnorm**2 - self.dampsq * self.xxnorm
        self.r1norm = self.ncp.sqrt(self.ncp.abs(self.r1sq))
        self.cost.append(float(self.r1norm))
        if self.r1sq < 0:
            self.r1norm = -self.r1norm
        self.r2norm = self.rnorm.copy()

        # use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        self.test1 = self.rnorm / self.bnorm
        self.test2 = self.arnorm / self.arnorm0
        self.test3 = 1.0 / self.acond
        t1 = self.test1 / (1.0 + self.anorm * self.xnorm / self.bnorm)
        self.rtol = self.btol + self.atol * self.anorm * self.xnorm / self.bnorm

        # set reason for termination.
        # The following tests guard against extremely small values of
        # atol, btol  or ctol. The effect is equivalent to the normal tests
        # using atol = eps,  btol = eps, conlim = 1/eps.
        if self.iiter >= self.niter:
            self.istop = 7
        if 1 + self.test3 <= 1:
            self.istop = 6
        if 1 + self.test2 <= 1:
            self.istop = 5
        if 1 + t1 <= 1:
            self.istop = 4

        # allow for tolerances set by the user.
        if self.test3 <= self.ctol:
            self.istop = 3
        if self.test2 <= self.atol:
            self.istop = 2
        if self.test1 <= self.rtol:
            self.istop = 1

        self.iiter += 1
        if show:
            self._print_step(x)
        return x

    def run(
        self,
        x: NDArray,
        niter: Optional[int] = None,
        show: bool = False,
        itershow: List[int] = [10, 10, 10],
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of LSQR
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display iterations log
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        niter = self.niter if niter is None else niter
        while self.iiter < niter and self.istop == 0:
            showstep = (
                True
                if show
                and (
                    self.niter <= 40
                    or self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                    or self.test3 <= 2 * self.ctol
                    or self.test2 <= 10 * self.atol
                    or self.test1 <= 10 * self.rtol
                    or self.istop != 0
                )
                else False
            )
            x = self.step(x, showstep)
            self.callback(x)
        return x

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        self.cost = np.array(self.cost)
        if show:
            self._print_finalize()

    def solve(
        self,
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
    ) -> Tuple[NDArray, int, float, float, float, float, float, float, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
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
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

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

        """
        x = self.setup(
            y=y,
            x0=x0,
            damp=damp,
            atol=atol,
            btol=btol,
            conlim=conlim,
            niter=niter,
            calc_var=calc_var,
            show=show,
        )
        x = self.run(x, niter=niter, show=show, itershow=itershow)
        self.finalize(show)
        return (
            x,
            self.istop,
            self.iiter,
            self.r1norm,
            self.r2norm,
            self.anorm,
            self.acond,
            self.arnorm,
            self.xnorm,
            self.var,
            self.cost,
        )
