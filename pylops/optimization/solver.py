import time

import numpy as np

from pylops.optimization.basesolver import Solver
from pylops.utils.backend import get_array_module


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
    Solve the :math:`\mathbf{y} = \mathbf{Opx}` problem using conjugate gradient
    iterations [1]_.

    .. [1] Hestenes, M R., Stiefel, E., “Methods of Conjugate Gradients for Solving
       Linear Systems”, Journal of Research of the National Bureau of Standards.
       vol. 49. 1952.

    """

    def _print_setup(self, xcomplex=False):
        self._print_solver()

        if self.niter is not None:
            strpar = f"tol = {self.tol:10e}\tniter = {self.niter}"
        else:
            strpar = f"tol = {self.tol:10e}"
        print(strpar)
        print("-----------------------------------------------------------")
        if not xcomplex:
            head1 = "    Itn           x[0]              r2norm"
        else:
            head1 = "    Itn              x[0]                  r2norm"
        print(head1)

    def _print_step(self, x):
        strx = f"{x[0]:1.2e}        " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = f"{self.iiter:6g}        " + strx + f"{self.cost[self.iiter]:11.4e}"
        print(msg)

    def setup(self, y=None, x0=None, niter=None, tol=1e-4, show=False):
        """Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`
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
        self.tol = tol
        self.niter = niter
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

    def step(self, x, show=False):
        """Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of CG
        show : :obj:`bool`, optional
            Display iteration log

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

    def finalize(self, show=False):
        """Finalize solver

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

    def solve(self, y, x0=None, niter=10, tol=1e-4, show=False, itershow=[10, 10, 10]):
        """Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess
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
        cost : :obj:`numpy.ndarray`, optional
            History of the L2 norm of the residual

        """
        x = self.setup(y=y, x0=x0, niter=niter, tol=tol, show=show)
        while self.iiter < niter and self.kold > self.tol:
            show = (
                True
                if self.iiter < itershow[0]
                or niter - self.iiter < itershow[1]
                or self.iiter % itershow[2] == 0
                else False
            )
            x = self.step(x, show)
            self.callback(x)
        self.finalize(show)
        return x, self.iiter, self.cost


def cg(Op, y, x0, niter=10, tol=1e-4, show=False, itershow=[10, 10, 10], callback=None):
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
    damp : :obj:`float`, optional
        *Deprecated*, will be removed in v2.0.0
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
    See :class:`pylops.optimization.solver.CG`

    """
    cgsolve = CG(Op)
    if callback is not None:
        cgsolve.callback = callback
    x, iiter, cost = cgsolve.solve(
        y=y, x0=x0, tol=tol, niter=niter, show=show, itershow=itershow
    )
    return x, iiter, cost


class CGLS(Solver):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``Op`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert of size :math:`[N \times N]`

    Notes
    -----
    Minimize the following functional using conjugate gradient iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Opx} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    """

    def _print_setup(self, xcomplex=False):
        self._print_solver()

        if self.niter is not None:
            strpar = (
                f"damp = {self.damp:10e}\ttol = {self.tol:10e}\tniter = {self.niter}"
            )
        else:
            strpar = f"damp = {self.damp:10e}\ttol = {self.tol:10e}\t"
        print(strpar)
        print("-----------------------------------------------------------")
        if not xcomplex:
            head1 = "    Itn          x[0]              r1norm         r2norm"
        else:
            head1 = "    Itn             x[0]             r1norm         r2norm"
        print(head1)

    def _print_step(self, x):
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = (
            f"{self.iiter:6g}       "
            + strx
            + f"{self.cost[self.iiter]:11.4e}    {self.cost1[self.iiter]:11.4e}"
        )
        print(msg)

    def setup(self, y=None, x0=None, niter=None, damp=0.0, tol=1e-4, show=False):
        """Setup solver

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
        self.cost.append(self.ncp.linalg.norm(self.s))
        self.cost1.append(
            self.ncp.sqrt(self.cost[0] ** 2 + damp * self.ncp.abs(x.dot(x.conj())))
        )
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x))
        return x

    def step(self, x, show=False):
        """Run one step of solver

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
        self.cost.append(self.ncp.linalg.norm(self.s))
        self.cost1.append(
            self.ncp.sqrt(
                self.cost[self.iiter] ** 2 + self.damp * self.ncp.abs(x.dot(x.conj()))
            )
        )
        if show:
            self._print_step(x)
        return x

    def finalize(self, show=False):
        """Finalize solver

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
            self._print_finalize()
        self.cost = np.array(self.cost)

    def solve(
        self,
        y,
        x0=None,
        niter=10,
        damp=0.0,
        tol=1e-4,
        show=False,
        itershow=[10, 10, 10],
    ):
        """Run entire solver

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
            Display setup log
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
            :math:`\mathbf{d} = \mathbf{Op}\,\mathbf{x}`

            ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
            problem
        iit : :obj:`int`
            Iteration number upon termination
        r1norm : :obj:`float`
            :math:`||\mathbf{r}||_2`, where
            :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\,\mathbf{x}`
        r2norm : :obj:`float`
            :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
            \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
            Equal to ``r1norm`` if :math:`\epsilon=0`
        cost : :obj:`numpy.ndarray`, optional
            History of r1norm through iterations


        """
        x = self.setup(y=y, x0=x0, niter=niter, damp=damp, tol=tol, show=show)
        while self.iiter < niter and self.kold > self.tol:
            show = (
                True
                if self.iiter < itershow[0]
                or niter - self.iiter < itershow[1]
                or self.iiter % itershow[2] == 0
                else False
            )
            x = self.step(x, show)
            self.callback(x)
        self.finalize(show)
        return x, self.istop, self.iiter, self.r1norm, self.r2norm, self.cost


def cgls(
    Op,
    y,
    x0,
    niter=10,
    damp=0.0,
    tol=1e-4,
    show=False,
    itershow=[10, 10, 10],
    callback=None,
):
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
        :math:`\mathbf{d} = \mathbf{Op}\,\mathbf{x}`

        ``2`` means :math:`\mathbf{x}` approximately solves the least-squares
        problem
    iit : :obj:`int`
        Iteration number upon termination
    r1norm : :obj:`float`
        :math:`||\mathbf{r}||_2`, where
        :math:`\mathbf{r} = \mathbf{d} - \mathbf{Op}\,\mathbf{x}`
    r2norm : :obj:`float`
        :math:`\sqrt{\mathbf{r}^T\mathbf{r}  +
        \epsilon^2 \mathbf{x}^T\mathbf{x}}`.
        Equal to ``r1norm`` if :math:`\epsilon=0`
    cost : :obj:`numpy.ndarray`, optional
        History of r1norm through iterations

    Notes
    -----
    See :class:`pylops.optimization.solver.CGLS`


    """
    cgsolve = CGLS(Op)
    if callback is not None:
        cgsolve.callback = callback
    x, istop, iiter, r1norm, r2norm, cost = cgsolve.solve(
        y=y, x0=x0, tol=tol, niter=niter, show=show, itershow=itershow
    )
    return x, istop, iiter, r1norm, r2norm, cost


def lsqr(
    Op,
    y,
    x0,
    damp=0.0,
    atol=1e-08,
    btol=1e-08,
    conlim=100000000.0,
    niter=10,
    calc_var=True,
    show=False,
    callback=None,
):
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
        Initial guess
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
    Minimize the following functional using LSQR iterations [1]_:

    .. math::
        J = || \mathbf{y} -  \mathbf{Op}\,\mathbf{x} ||_2^2 +
        \epsilon^2 || \mathbf{x} ||_2^2

    where :math:`\epsilon` is the damping coefficient.

    .. [1] Paige, C. C., and Saunders, M. A. "LSQR: An algorithm for sparse
        linear equations and sparse least squares", ACM TOMS, vol. 8, pp. 43-71,
        1982.

    """
    # Return messages.
    msg = (
        "The exact solution is x = 0                               ",
        "Opx - b is small enough, given atol, btol                  ",
        "The least-squares solution is good enough, given atol     ",
        "The estimate of cond(Opbar) has exceeded conlim            ",
        "Opx - b is small enough for this machine                   ",
        "The least-squares solution is good enough for this machine",
        "Cond(Opbar) seems to be too large for this machine         ",
        "The iteration limit has been reached                      ",
    )

    ncp = get_array_module(y)
    m, n = Op.shape

    var = None
    if calc_var:
        var = ncp.zeros(n)

    if show:
        tstart = time.time()
        print("LSQR")
        print("-------------------------------------------------")
        str1 = f"The Operator Op has {m} rows and {n} cols"
        str2 = f"damp = {damp:20.14e}     calc_var = {calc_var:6g}"
        str3 = f"atol = {atol:8.2e}                 conlim = {conlim:8.2e}"
        str4 = f"btol = {btol:8.2e}                 niter = {niter:8g}"
        print(str1)
        print(str2)
        print(str3)
        print(str4)
        print("-------------------------------------------------")

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1.0 / conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # set up the first vectors u and v for the bidiagonalization.
    # These satisfy beta*u=b-Op(x0), alfa*v=Op'u
    if x0 is None:
        x = ncp.zeros(Op.shape[1], dtype=y.dtype)
        u = y.copy()
    else:
        x = x0.copy()
        u = y - Op.matvec(x0)
    alfa = 0.0
    beta = ncp.linalg.norm(u)
    if beta > 0.0:
        u = u / beta
        v = Op.rmatvec(u)
        alfa = ncp.linalg.norm(v)
        if alfa > 0:
            v = v / alfa
            w = v.copy()

    arnorm = alfa * beta
    if arnorm == 0:
        print(" ")
        print("LSQR finished")
        print(msg[istop])
        return x, istop, itn, 0, 0, anorm, acond, arnorm, xnorm, var
    arnorm0 = arnorm

    rhobar = alfa
    phibar = beta
    bnorm = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm
    cost = np.zeros(niter + 1)
    cost[0] = rnorm
    head1 = "   Itn      x[0]       r1norm     r2norm "
    head2 = " Compatible   LS      Norm A   Cond A"

    if show:
        print(" ")
        print(head1 + head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f"{itn:6g} {x[0]:12.5e}"
        str2 = f" {r1norm:10.3e} {r2norm:10.3e}"
        str3 = f"  {test1:8.1e} {test2:8.1e}"
        print(str1 + str2 + str3)

    # main iteration loop
    while itn < niter:
        itn = itn + 1
        # perform the next step of the bidiagonalization to obtain the
        # next beta, u, alfa, v. These satisfy the relations
        # beta*u = Op*v - alfa*u,
        # alfa*v = Op'*u - beta*v'
        u = Op.matvec(v) - alfa * u
        beta = ncp.linalg.norm(u)
        if beta > 0:
            u = u / beta
            anorm = np.linalg.norm([anorm, alfa, beta, damp])
            v = Op.rmatvec(u) - beta * v
            alfa = ncp.linalg.norm(v)
            if alfa > 0:
                v = v / alfa

        # use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = np.linalg.norm([rhobar, damp])
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        rho = np.linalg.norm([rhobar1, beta])
        cs = rhobar1 / rho
        sn = beta / rho
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = w / rho
        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + ncp.linalg.norm(dk) ** 2
        if calc_var:
            var = var + ncp.dot(dk, dk)

        # use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = ncp.sqrt(xxnorm + zbar**2)
        gamma = np.linalg.norm([gambar, theta])
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2.0

        # test for convergence. First, estimate the condition of the matrix
        # Opbar, and the norms of rbar and Opbar'rbar
        acond = anorm * ncp.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = ncp.sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # distinguish between r1norm = ||b - Ax|| and
        # r2norm = sqrt(r1norm^2 + damp^2*||x||^2).
        # Estimate r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = ncp.sqrt(ncp.abs(r1sq))
        cost[itn] = r1norm
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm.copy()

        # use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / arnorm0
        test3 = 1.0 / acond
        t1 = test1 / (1.0 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # set reason for termination.
        # The following tests guard against extremely small values of
        # atol, btol  or ctol. The effect is equivalent to the normal tests
        # using atol = eps,  btol = eps, conlim = 1/eps.
        if itn >= niter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # run callback
        if callback is not None:
            callback(x)

        # print status
        if show:
            if (
                n <= 40
                or itn <= 10
                or itn >= niter - 10
                or itn % 10 == 0
                or test3 <= 2 * ctol
                or test2 <= 10 * atol
                or test1 <= 10 * rtol
                or istop != 0
            ):
                str1 = f"{itn:6g} {x[0]:12.5e}"
                str2 = f" {r1norm:10.3e} {r2norm:10.3e}"
                str3 = f"  {test1:8.1e} {test2:8.1e}"
                str4 = f" {anorm:8.1e} {acond:8.1e}"
                print(str1 + str2 + str3 + str4)
        if istop > 0:
            break

    # Print the stopping condition.
    if show:
        print(" ")
        print(f"LSQR finished, {msg[istop]}")
        print(" ")
        str1 = f"istop ={istop:8g}   r1norm ={r1norm:8.1e}"
        str2 = f"anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}"
        str3 = f"itn   ={itn:8g}   r2norm ={r2norm:8.1e}"
        str4 = f"acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}"
        str5 = f"Total time (s) = {time.time() - tstart:.2f}"
        print(str1 + "   " + str2)
        print(str3 + "   " + str4)
        print(str5)
        print(
            "-----------------------------------------------------------------------\n"
        )

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var, cost[:itn]
