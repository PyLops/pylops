from pylops.optimization.sparsityc import FISTA, IRLS, ISTA, OMP, SPGL1, SplitBregman
from pylops.utils.decorators import add_ndarray_support_to_solver


def irls(
    Op,
    y,
    x0=None,
    nouter=10,
    threshR=False,
    epsR=1e-10,
    epsI=1e-10,
    tolIRLS=1e-10,
    kind="data",
    show=False,
    itershow=[10, 10, 10],
    callback=None,
    callbacks=None,
    **kwargs_solver,
):
    r"""Iteratively reweighted least squares.

    Solve an optimization problem with :math:`L_1` cost function (data IRLS)
    or :math:`L_1` regularization term (model IRLS) given the operator ``Op``
    and data ``y``.

    In the *data IRLS*, the cost function is minimized by iteratively solving a
    weighted least squares problem with the weight at iteration :math:`i`
    being based on the data residual at iteration :math:`i-1`. This IRLS solver
    is robust to *outliers* since the L1 norm given less weight to large
    residuals than L2 norm does.

    Similarly in the *model IRLS*, the weight at at iteration :math:`i`
    is based on the model at iteration :math:`i-1`. This IRLS solver inverts
    for a sparse model vector.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    data : :obj:`numpy.ndarray`
        Data
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    nouter : :obj:`int`, optional
        Number of outer iterations
    threshR : :obj:`bool`, optional
        Apply thresholding in creation of weight (``True``)
        or damping (``False``)
    epsR : :obj:`float`, optional
        Damping to be applied to residuals for weighting term
    espI : :obj:`float`, optional
        Tikhonov damping
    tolIRLS : :obj:`float`, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tolIRLS``
    kind : :obj:`str`, optional
        Kind of solver (``data`` or ``model``)
    show : :obj:`bool`, optional
        Display logs
    itershow : :obj:`list`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    **kwargs_solver
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.cg` solver for data IRLS and
        :py:func:`scipy.sparse.linalg.lsqr` solver for model IRLS when using
        numpy data(or :py:func:`pylops.optimization.solver.cg` and
        :py:func:`pylops.optimization.solver.cgls` when using cupy data)

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    nouter : :obj:`int`
        Number of effective outer iterations

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.IRLS`

    """
    irlssolve = IRLS(Op)
    if callback is not None:
        irlssolve.callback = callback
    x, nouter, = irlssolve.solve(
        y,
        x0=x0,
        nouter=nouter,
        threshR=threshR,
        epsR=epsR,
        epsI=epsI,
        tolIRLS=tolIRLS,
        kind=kind,
        show=show,
        itershow=itershow,
        **kwargs_solver,
    )
    return x, nouter


def omp(
    Op,
    y,
    niter_outer=10,
    niter_inner=40,
    sigma=1e-4,
    normalizecols=False,
    show=False,
    itershow=[10, 10, 10],
    callback=None,
):
    r"""Orthogonal Matching Pursuit (OMP).

    Solve an optimization problem with :math:`L^0` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data
    niter_outer : :obj:`int`, optional
        Number of iterations of outer loop
    niter_inner : :obj:`int`, optional
        Number of iterations of inner loop. By choosing ``niter_inner=0``, the
        Matching Pursuit (MP) algorithm is implemented.
    sigma : :obj:`list`
        Maximum :math:`L_2` norm of residual. When smaller stop iterations.
    normalizecols : :obj:`list`, optional
        Normalize columns (``True``) or not (``False``). Note that this can be
        expensive as it requires applying the forward operator
        :math:`n_{cols}` times to unit vectors (i.e., containing 1 at
        position j and zero otherwise); use only when the columns of the
        operator are expected to have highly varying norms.
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
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter_outer : :obj:`int`
        Number of effective outer iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    See Also
    --------
    ISTA: Iterative Shrinkage-Thresholding Algorithm (ISTA).
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.OMP`

    """
    ompsolve = OMP(Op)
    if callback is not None:
        ompsolve.callback = callback
    x, niter_outer, cost = ompsolve.solve(
        y,
        niter_outer=niter_outer,
        niter_inner=niter_inner,
        sigma=sigma,
        normalizecols=normalizecols,
        show=show,
        itershow=itershow,
    )
    return x, niter_outer, cost


def ista(
    Op,
    y,
    x0=None,
    niter=10,
    SOp=None,
    eps=0.1,
    alpha=None,
    eigsiter=None,
    eigstol=0,
    tol=1e-10,
    threshkind="soft",
    perc=None,
    decay=None,
    monitorres=False,
    show=False,
    itershow=[10, 10, 10],
    callback=None,
):
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``. The operator
    can be real or complex, and should ideally be either square :math:`N=M`
    or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data of size :math:`[N \times 1]`
    x0: :obj:`numpy.ndarray`, optional
        Initial guess
    niter : :obj:`int`
        Number of iterations
    SOp : :obj:`pylops.LinearOperator`, optional
        Regularization operator (use when solving the analysis problem)
    eps : :obj:`float`, optional
        Sparsity damping
    alpha : :obj:`float`, optional
        Step size. To guarantee convergence, ensure
        :math:`\alpha \le 1/\lambda_\text{max}`, where :math:`\lambda_\text{max}`
        is the largest eigenvalue of :math:`\mathbf{Op}^H\mathbf{Op}`.
        If ``None``, the maximum eigenvalue is estimated and the optimal step size
        is chosen as :math:`1/\lambda_\text{max}`. If provided, the
        convergence criterion will not be checked internally.
    eigsiter : :obj:`float`, optional
        Number of iterations for eigenvalue estimation if ``alpha=None``
    eigstol : :obj:`float`, optional
        Tolerance for eigenvalue estimation if ``alpha=None``
    tol : :obj:`float`, optional
        Tolerance. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half', 'hard-percentile',
        'soft-percentile', or 'half-percentile' - 'soft' used as default)
    perc : :obj:`float`, optional
        Percentile, as percentage of values to be kept by thresholding (to be
        provided when thresholding is soft-percentile or half-percentile)
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    monitorres : :obj:`bool`, optional
        Monitor that residual is decreasing
    show : :obj:`bool`, optional
        Display logs
    itershow : :obj:`list`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half, soft-percentile,
        or half-percentile
    ValueError
        If ``perc=None`` when ``threshkind`` is soft-percentile or
        half-percentile
    ValueError
        If ``monitorres=True`` and residual increases

    See Also
    --------
    OMP: Orthogonal Matching Pursuit (OMP).
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.ISTA`

    """
    istasolve = ISTA(Op)
    if callback is not None:
        istasolve.callback = callback
    x, iiter, cost = istasolve.solve(
        y=y,
        x0=x0,
        niter=niter,
        SOp=SOp,
        eps=eps,
        alpha=alpha,
        eigsiter=eigsiter,
        eigstol=eigstol,
        tol=tol,
        threshkind=threshkind,
        perc=perc,
        decay=decay,
        monitorres=monitorres,
        show=show,
        itershow=itershow,
    )
    return x, iiter, cost


def fista(
    Op,
    y,
    x0=None,
    niter=10,
    SOp=None,
    eps=0.1,
    alpha=None,
    eigsiter=None,
    eigstol=0,
    tol=1e-10,
    threshkind="soft",
    perc=None,
    decay=None,
    monitorres=False,
    show=False,
    itershow=[10, 10, 10],
    callback=None,
):
    r"""Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``.
    The operator can be real or complex, and should ideally be either square
    :math:`N=M` or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data
    x0: :obj:`numpy.ndarray`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    SOp : :obj:`pylops.LinearOperator`, optional
        Regularization operator (use when solving the analysis problem)
    eps : :obj:`float`, optional
        Sparsity damping
    alpha : :obj:`float`, optional
        Step size. To guarantee convergence, ensure
        :math:`\alpha \le 1/\lambda_\text{max}`, where :math:`\lambda_\text{max}`
        is the largest eigenvalue of :math:`\mathbf{Op}^H\mathbf{Op}`.
        If ``None``, the maximum eigenvalue is estimated and the optimal step size
        is chosen as :math:`1/\lambda_\text{max}`. If provided, the
        convergence criterion will not be checked internally.
    eigsiter : :obj:`int`, optional
        Number of iterations for eigenvalue estimation if ``alpha=None``
    eigstol : :obj:`float`, optional
        Tolerance for eigenvalue estimation if ``alpha=None``
    tol : :obj:`float`, optional
        Tolerance. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half', 'soft-percentile', or
        'half-percentile' - 'soft' used as default)
    perc : :obj:`float`, optional
        Percentile, as percentage of values to be kept by thresholding (to be
        provided when thresholding is soft-percentile or half-percentile)
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    monitorres : :obj:`bool`, optional
            Monitor that residual is decreasing
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
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    Raises
    ------
    NotImplementedError
        If ``threshkind`` is different from hard, soft, half, soft-percentile,
        or half-percentile
    ValueError
        If ``perc=None`` when ``threshkind`` is soft-percentile or
        half-percentile

    See Also
    --------
    OMP: Orthogonal Matching Pursuit (OMP).
    ISTA: Iterative Shrinkage-Thresholding Algorithm (ISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.FISTA`

    """
    fistasolve = FISTA(Op)
    if callback is not None:
        fistasolve.callback = callback
    x, iiter, cost = fistasolve.solve(
        y=y,
        x0=x0,
        niter=niter,
        SOp=SOp,
        eps=eps,
        alpha=alpha,
        eigsiter=eigsiter,
        eigstol=eigstol,
        tol=tol,
        threshkind=threshkind,
        perc=perc,
        decay=decay,
        monitorres=monitorres,
        show=show,
        itershow=itershow,
    )
    return x, iiter, cost


@add_ndarray_support_to_solver
def spgl1(Op, y, x0=None, SOp=None, tau=0, sigma=0, show=False, **kwargs_spgl1):
    r"""Spectral Projected-Gradient for L1 norm.

    Solve a constrained system of equations given the operator ``Op``
    and a sparsyfing transform ``SOp`` aiming to retrive a model that
    is sparse in the sparsyfing domain.

    This is a simple wrapper to :py:func:`spgl1.spgl1`
    which is a porting of the well-known
    `SPGL1 <https://www.cs.ubc.ca/~mpf/spgl1/>`_ MATLAB solver into Python.
    In order to be able to use this solver you need to have installed the
    ``spgl1`` library.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    SOp : :obj:`pylops.LinearOperator`, optional
        Sparsifying transform
    tau : :obj:`float`, optional
        Non-negative LASSO scalar. If different from ``0``,
        SPGL1 will solve LASSO problem
    sigma : :obj:`list`, optional
        BPDN scalar. If different from ``0``,
        SPGL1 will solve BPDN problem
    **kwargs_spgl1
        Arbitrary keyword arguments for
        :py:func:`spgl1.spgl1` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model in original domain.
    pinv : :obj:`numpy.ndarray`
        Inverted model in sparse domain.
    info : :obj:`dict`
        Dictionary with the following information:

        - ``tau``, final value of tau (see sigma above)

        - ``rnorm``, two-norm of the optimal residual

        - ``rgap``, relative duality gap (an optimality measure)

        - ``gnorm``, Lagrange multiplier of (LASSO)

        - ``stat``, final status of solver
           * ``1``: found a BPDN solution,
           * ``2``: found a BP solution; exit based on small gradient,
           * ``3``: found a BP solution; exit based on small residual,
           * ``4``: found a LASSO solution,
           * ``5``: error, too many iterations,
           * ``6``: error, linesearch failed,
           * ``7``: error, found suboptimal BP solution,
           * ``8``: error, too many matrix-vector products.

        - ``niters``, number of iterations

        - ``nProdA``, number of multiplications with A

        - ``nProdAt``, number of multiplications with A'

        - ``n_newton``, number of Newton steps

        - ``time_project``, projection time (seconds)

        - ``time_matprod``, matrix-vector multiplications time (seconds)

        - ``time_total``, total solution time (seconds)

        - ``niters_lsqr``, number of lsqr iterations (if ``subspace_min=True``)

        - ``xnorm1``, L1-norm model solution history through iterations

        - ``rnorm2``, L2-norm residual history through iterations

        - ``lambdaa``, Lagrange multiplier history through iterations

    Raises
    ------
    ModuleNotFoundError
        If the ``spgl1`` library is not installed

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.SPGL1`

    """
    spgl1solve = SPGL1(Op)
    xinv, pinv, info = spgl1solve.solve(
        y,
        x0=x0,
        SOp=SOp,
        tau=tau,
        sigma=sigma,
        show=show,
        **kwargs_spgl1,
    )
    return xinv, pinv, info


def splitbregman(
    Op,
    y,
    RegsL1,
    x0=None,
    niter_outer=3,
    niter_inner=5,
    RegsL2=None,
    dataregsL2=None,
    mu=1.0,
    epsRL1s=None,
    epsRL2s=None,
    tol=1e-10,
    tau=1.0,
    restart=False,
    show=False,
    itershow=[10, 10, 10],
    show_inner=False,
    callback=None,
    **kwargs_lsqr,
):
    r"""Split Bregman for mixed L2-L1 norms.

    Solve an unconstrained system of equations with mixed :math:`L_2` and :math:`L_1`
    regularization terms given the operator ``Op``, a list of :math:`L_1`
    regularization terms ``RegsL1``, and an optional list of :math:`L_2`
    regularization terms ``RegsL2``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    y : :obj:`numpy.ndarray`
        Data
    RegsL1 : :obj:`list`
        :math:`L_1` regularization operators
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    niter_outer : :obj:`int`
        Number of iterations of outer loop
    niter_inner : :obj:`int`
        Number of iterations of inner loop of first step of the Split Bregman
        algorithm. A small number of iterations is generally sufficient and
        for many applications optimal efficiency is obtained when only one
        iteration is performed.
    RegsL2 : :obj:`list`
        Additional :math:`L_2` regularization operators
        (if ``None``, :math:`L_2` regularization is not added to the problem)
    dataregsL2 : :obj:`list`, optional
        :math:`L_2` Regularization data (must have the same number of elements
        of ``RegsL2`` or equal to ``None`` to use a zero data for every
        regularization operator in ``RegsL2``)
    mu : :obj:`float`, optional
         Data term damping
    epsRL1s : :obj:`list`
         :math:`L_1` Regularization dampings (must have the same number of elements
         as ``RegsL1``)
    epsRL2s : :obj:`list`
         :math:`L_2` Regularization dampings (must have the same number of elements
         as ``RegsL2``)
    tol : :obj:`float`, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    tau : :obj:`float`, optional
        Scaling factor in the Bregman update (must be close to 1)
    restart : :obj:`bool`, optional
        The unconstrained inverse problem in inner loop is initialized with
        the initial guess (``True``) or with the last estimate (``False``)
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
    show_inner : :obj:`bool`, optional
        Display inner iteration logs of lsqr
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    **kwargs_lsqr
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver used to solve the first
        subproblem in the first step of the Split Bregman algorithm.

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    itn_out : :obj:`int`
        Iteration number of outer loop upon termination
    cost : :obj:`numpy.ndarray`, optional
            History of cost function through iterations

    Notes
    -----
    See :class:`pylops.optimization.sparsityc.SplitBregman`

    """
    sbsolve = SplitBregman(Op)
    if callback is not None:
        sbsolve.callback = callback
    xinv, itn_out, cost = sbsolve.solve(
        y,
        RegsL1,
        x0=x0,
        niter_outer=niter_outer,
        niter_inner=niter_inner,
        RegsL2=RegsL2,
        dataregsL2=dataregsL2,
        mu=mu,
        epsRL1s=epsRL1s,
        epsRL2s=epsRL2s,
        tol=tol,
        tau=tau,
        restart=restart,
        show=show,
        itershow=itershow,
        show_inner=show_inner,
        **kwargs_lsqr,
    )
    return xinv, itn_out, cost
