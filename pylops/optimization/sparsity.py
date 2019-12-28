import time
import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import LinearOperator
from pylops.basicoperators import Diagonal
from pylops.optimization.leastsquares import NormalEquationsInversion, \
    RegularizedInversion

try:
    from spgl1 import spgl1
except ModuleNotFoundError:
    spgl1 = None
    spgl1_message = 'Spgl1 not installed. ' \
                    'Run "pip install spgl1".'
except Exception as e:
    spgl1 = None
    spgl1_message = 'Failed to import spgl1 (error:%s).' % e


def _softthreshold(x, thresh):
    r"""Soft thresholding.

    Applies soft thresholding (proximity operator for :math:`||\mathbf{x}||_1`)
    to vector ``x``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold
    """
    return np.maximum(np.abs(x)-thresh, 0.)*np.sign(x)

def _shrinkage(x, thresh):
    r"""Shrinkage.

    Applies shrinkage to vector ``x``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold
    """
    xabs = np.abs(x)
    return x/(xabs+1e-10) * np.maximum(xabs - thresh, 0)


def IRLS(Op, data, nouter, threshR=False, epsR=1e-10,
         epsI=1e-10, x0=None, tolIRLS=1e-10,
         returnhistory=False, **kwargs_cg):
    r"""Iteratively reweighted least squares.

    Solve an optimization problem with :math:`L1` cost function given the
    operator ``Op`` and data ``y``. The cost function is minimized by
    iteratively solving a weighted least squares problem with the weight at
    iteration :math:`i` being based on the data residual at iteration
    :math:`i+1`.

    The IRLS solver is robust to *outliers* since the L1 norm given less
    weight to large residuals than L2 norm does.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    data : :obj:`numpy.ndarray`
        Data
    nouter : :obj:`int`
        Number of outer iterations
    threshR : :obj:`bool`, optional
        Apply thresholding in creation of weight (``True``)
        or damping (``False``)
    epsR : :obj:`float`, optional
        Damping to be applied to residuals for weighting term
    espI : :obj:`float`, optional
        Tikhonov damping
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    tolIRLS : :obj:`float`, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tolIRLS``
    returnhistory : :obj:`bool`, optional
        Return history of inverted model for each outer iteration of IRLS
    **kwargs_cg
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.cg` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    nouter : :obj:`int`
        Number of effective outer iterations
    xinv_hist : :obj:`numpy.ndarray`, optional
        History of inverted model
    rw_hist : :obj:`numpy.ndarray`, optional
        History of weights

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = ||\mathbf{d} - \mathbf{Op} \mathbf{x}||_1

    by a set of outer iterations which require to repeateadly solve a
    weighted least squares problem of the form:

    .. math::
        \mathbf{x}^{(i+1)} = \operatorname*{arg\,min}_\mathbf{x} ||\mathbf{d} -
        \mathbf{Op} \mathbf{x}||_{2, \mathbf{R}^{(i)}} +
        \epsilon_I^2 ||\mathbf{x}||

    where :math:`\mathbf{R}^{(i)}` is a diagonal weight matrix
    whose diagonal elements at iteration :math:`i` are equal to the absolute
    inverses of the residual vector :math:`\mathbf{r}^{(i)} =
    \mathbf{y} - \mathbf{Op} \mathbf{x}^{(i)}` at iteration :math:`i`.
    More specifically the j-th element of the diagonal of
    :math:`\mathbf{R}^{(i)}` is

    .. math::
        R^{(i)}_{j,j} = \frac{1}{|r^{(i)}_j|+\epsilon_R}

    or

    .. math::
        R^{(i)}_{j,j} = \frac{1}{max(|r^{(i)}_j|, \epsilon_R)}

    depending on the choice ``threshR``. In either case,
    :math:`\epsilon_R` is the user-defined stabilization/thresholding
    factor [1]_.

    .. [1] https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares

    """
    if x0 is not None:
        data = data - Op * x0
    if returnhistory:
        xinv_hist = np.zeros((nouter+1, Op.shape[1]))
        rw_hist = np.zeros((nouter+1, Op.shape[0]))

    # first iteration (unweighted least-squares)
    xinv = NormalEquationsInversion(Op, None, data, epsI=epsI,
                                    returninfo=False,
                                    **kwargs_cg)
    r = data-Op*xinv
    if returnhistory:
        xinv_hist[0] = xinv
    for iiter in range(nouter):
        # other iterations (weighted least-squares)
        xinvold = xinv.copy()
        if threshR:
            rw = 1./np.maximum(np.abs(r), epsR)
        else:
            rw = 1./(np.abs(r)+epsR)
        rw = rw / rw.max()
        R = Diagonal(rw)
        xinv = NormalEquationsInversion(Op, [], data, Weight=R,
                                        epsI=epsI,
                                        returninfo=False,
                                        **kwargs_cg)
        r = data-Op*xinv
        # save history
        if returnhistory:
            rw_hist[iiter] = rw
            xinv_hist[iiter+1] = xinv
        # check tolerance
        if np.linalg.norm(xinv - xinvold) < tolIRLS:
            nouter = iiter
            break

    # adding initial guess
    if x0 is not None:
        xinv = x0 + xinv
        if returnhistory:
            xinv_hist = x0 + xinv_hist

    if returnhistory:
        return xinv, nouter, xinv_hist[:nouter+1], rw_hist[:nouter+1]
    else:
        return xinv, nouter


def OMP(Op, data, niter_outer=10, niter_inner=40, sigma=1e-4,
        normalizecols=False, show=False):
    r"""Orthogonal Matching Pursuit (OMP).

    Solve an optimization problem with :math:`L0` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    data : :obj:`numpy.ndarray`
        Data
    niter_outer : :obj:`int`
        Number of iterations of outer loop
    niter_inner : :obj:`int`
        Number of iterations of inner loop
    sigma : :obj:`list`
        Maximum L2 norm of residual. When smaller stop iterations.
    normalizecols : :obj:`list`
        Normalize columns (``True``) or not (``False``). Note that this can be
        expensive as it requires applying the forward operator
        :math:`n_{cols}` times to unit vectors (i.e., containing 1 at
        position j and zero otherwise); use only when the columns of the
        operator are expected to have highly varying norms.
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    iiter : :obj:`int`
        Number of effective outer iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    See Also
    --------
    ISTA: Iterative Soft Thresholding Algorithm (ISTA).
    FISTA: Fast Iterative Soft Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
            ||\mathbf{x}||_0 \quad  subj. to \quad
            ||\mathbf{Op}\mathbf{x}-\mathbf{b}||_2 <= \sigma,

    using Orthogonal Matching Pursuit (OMP). This is a very
    simple iterative algorithm which applies the following step:

    .. math::
        \Lambda_k = \Lambda_{k-1} \cup \{ arg max_j
        |\mathbf{Op}_j^H \mathbf{r}_k| \} \\
        \mathbf{x}_k =  \{ arg min_{\mathbf{x}}
        ||\mathbf{Op}_{\Lambda_k} \mathbf{x} - \mathbf{b}||_2

    """
    Op = LinearOperator(Op)
    if show:
        tstart = time.time()
        print('OMP optimization\n'
              '-----------------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'sigma = %.2e\tniter_outer = %d\tniter_inner = %d\n'
              'normalization=%s' %
              (Op.shape[0], Op.shape[1], sigma, niter_outer,
               niter_inner, normalizecols))
    # find normalization factor for each column
    if normalizecols:
        ncols = Op.shape[1]
        norms = np.zeros(ncols)
        for icol in range(ncols):
            unit = np.zeros(ncols)
            unit[icol] = 1
            norms[icol] = np.linalg.norm(Op.matvec(unit))
    if show:
        print('-----------------------------------------------------------------')
        head1 = '    Itn           r2norm'
        print(head1)

    cols = []
    res = data.copy()
    cost = np.zeros(niter_outer + 1)
    cost[0] = np.linalg.norm(data)
    iiter = 0
    while iiter < niter_outer and cost[iiter] > sigma:
        cres = np.abs(Op.rmatvec(res))
        if normalizecols:
            cres = cres / norms
        # exclude columns already chosen by putting them negative
        if iiter > 0:
            cres[cols] = -1
        # choose column with max cres
        imax = np.argwhere(cres == np.max(cres)).ravel()
        nimax = len(imax)
        if nimax > 0:
            imax = imax[np.random.permutation(nimax)[0]]
        else:
            imax = imax[0]
        cols.append(imax)

        # estimate model for current set of columns
        Opcol = Op.apply_columns(cols)
        x = lsqr(Opcol, data, iter_lim=niter_inner)[0]
        res = data - Opcol.matvec(x)
        iiter += 1
        cost[iiter] = np.linalg.norm(res)
        if show:
            if iiter < 10 or niter_outer - iiter < 10 or iiter % 10 == 0:
                msg = '%6g        %12.5e' % (iiter + 1, cost[iiter])
                print(msg)
    xinv = np.zeros(Op.shape[1], dtype=Op.dtype)
    xinv[cols] = x
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (iiter, time.time() - tstart))
        print('-----------------------------------------------------------------\n')
    return xinv, iiter, cost


def ISTA(Op, data, niter, eps=0.1, alpha=None, eigsiter=None, eigstol=0,
         tol=1e-10, monitorres=False, returninfo=False, show=False):
    r"""Iterative Soft Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L1` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    data : :obj:`numpy.ndarray`
        Data
    niter : :obj:`int`
        Number of iterations
    eps : :obj:`float`, optional
        Sparsity damping
    alpha : :obj:`float`, optional
        Step size (:math:`\alpha \le 1/\lambda_{max}(\mathbf{Op}^H\mathbf{Op})`
        guarantees convergence. If ``None``, estimated to satisfy the
        condition, otherwise the condition will not be checked)
    eigsiter : :obj:`float`, optional
        Number of iterations for eigenvalue estimation if ``alpha=None``
    eigstol : :obj:`float`, optional
        Tolerance for eigenvalue estimation if ``alpha=None``
    tol : :obj:`float`, optional
        Tolerance. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    monitorres : :obj:`bool`, optional
        Monitor that residual is decreasing
    returninfo : :obj:`bool`, optional
        Return info of CG solver
    show : :obj:`bool`, optional
        Display iterations log

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
    ValueError
        If ``monitorres=True`` and residual increases

    See Also
    --------
    OMP: Orthogonal Matching Pursuit (OMP).
    FISTA: Fast Iterative Soft Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = ||\mathbf{d} - \mathbf{Op} \mathbf{x}||_2^2 +
            \epsilon ||\mathbf{x}||_1

    using the Iterative Soft Thresholding Algorithm (ISTA) [1]_. This is a very
    simple iterative algorithm which applies the following step:

    .. math::
        \mathbf{x}^{(i+1)} = soft (\mathbf{x}^{(i)} + \alpha \mathbf{Op}^H
        (\mathbf{d} - \mathbf{Op} \mathbf{x}^{(i)})), \epsilon \alpha /2)

    where :math:`\epsilon \alpha /2` is the
    threshold and :math:`soft()` is the so-called soft-thresholding rule.

    .. [1] Beck, A., and Teboulle, M., “A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems”, SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    if show:
        tstart = time.time()
        print('ISTA optimization\n'
              '-----------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'eps = %10e\ttol = %10e\tniter = %d' % (Op.shape[0],
                                                      Op.shape[1],
                                                      eps, tol, niter))
    # step size
    if alpha is None:
        if not isinstance(Op, LinearOperator):
            Op = LinearOperator(Op, explicit=False)
        # compute largest eigenvalues of Op^H * Op
        Op1 = LinearOperator(Op.H * Op, explicit=False)
        maxeig = np.abs(Op1.eigs(neigs=1, symmetric=True, niter=eigsiter,
                                 **dict(tol=eigstol, which='LM')))[0]
        alpha = 1./maxeig

    # define threshold
    thresh = eps*alpha*0.5

    if show:
        print('alpha = %10e\tthresh = %10e' % (alpha, thresh))
        print('-----------------------------------------------------------\n')
        head1 = '   Itn       x[0]        r2norm     r12norm     xupdate'
        print(head1)

    # initialize model and cost function
    xinv = np.zeros(Op.shape[1], dtype=Op.dtype)
    if monitorres:
        normresold = np.inf
    if returninfo:
        cost = np.zeros(niter+1)

    # iterate
    for iiter in range(niter):
        xinvold = xinv.copy()

        # compute residual
        res = data - Op.matvec(xinv)
        if monitorres:
            normres = np.linalg.norm(res)
            if  normres > normresold:
                raise ValueError('ISTA stopped at iteration %d due to '
                                 'residual increasing, consider modyfing '
                                 'eps and/or alpha...' % iiter)
            else:
                normresold = normres

        # compute gradient
        grad = alpha*Op.rmatvec(res)

        # update inverted model
        xinv_unthesh = xinv + grad
        xinv = _softthreshold(xinv_unthesh, thresh)

        # model update
        xupdate = np.linalg.norm(xinv - xinvold)

        if returninfo or show:
            costdata = 0.5 * np.linalg.norm(res) ** 2
            costreg = eps * np.linalg.norm(xinv, ord=1)
        if returninfo:
            cost[iiter] = costdata + costreg

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                msg = '%6g  %12.5e  %10.3e   %9.3e  %10.3e' % \
                      (iiter+1, xinv[0], costdata, costdata+costreg, xupdate)
                print(msg)

        # check tolerance
        if xupdate < tol:
            niter = iiter
            break

    # get values pre-threshold at locations where xinv is different from zero
    #xinv = np.where(xinv != 0, xinv_unthesh, xinv)
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (niter, time.time() - tstart))
        print('---------------------------------------------------------\n')
    if returninfo:
        return xinv, niter, cost[:niter]
    else:
        return xinv, niter


def FISTA(Op, data, niter, eps=0.1, alpha=None, eigsiter=None, eigstol=0,
          tol=1e-10, returninfo=False, show=False):
    r"""Fast Iterative Soft Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L1` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    data : :obj:`numpy.ndarray`
        Data
    niter : :obj:`int`
        Number of iterations
    eps : :obj:`float`, optional
        Sparsity damping
    alpha : :obj:`float`, optional
        Step size (:math:`\alpha \le 1/\lambda_{max}(\mathbf{Op}^H\mathbf{Op})`
        guarantees convergence. If ``None``, estimated to satisfy the
        condition, otherwise the condition will not be checked)
    eigsiter : :obj:`int`, optional
        Number of iterations for eigenvalue estimation if ``alpha=None``
    eigstol : :obj:`float`, optional
        Tolerance for eigenvalue estimation if ``alpha=None``
    tol : :obj:`float`, optional
        Tolerance. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    returninfo : :obj:`bool`, optional
        Return info of FISTA solver
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function

    See Also
    --------
    OMP: Orthogonal Matching Pursuit (OMP).
    ISTA: Iterative Soft Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = ||\mathbf{d} - \mathbf{Op} \mathbf{x}||_2^2 +
            \epsilon ||\mathbf{x}||_1

    using the Fast Iterative Soft Thresholding Algorithm (FISTA) [1]_. This is
    a modified version of ISTA solver with improved convergence properties and
    limitied additional computational cost.

    .. [1] Beck, A., and Teboulle, M., “A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems”, SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    if show:
        tstart = time.time()
        print('FISTA optimization\n'
              '-----------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'eps = %10e\ttol = %10e\tniter = %d' % (Op.shape[0],
                                                      Op.shape[1],
                                                      eps, tol, niter))
    # step size
    if alpha is None:
        if not isinstance(Op, LinearOperator):
            Op = LinearOperator(Op, explicit=False)
        # compute largest eigenvalues of Op^H * Op
        Op1 = LinearOperator(Op.H * Op, explicit=False)
        maxeig = np.abs(Op1.eigs(neigs=1, symmetric=True, niter=eigsiter,
                                 **dict(tol=eigstol, which='LM')))[0]
        alpha = 1./maxeig

    # define threshold
    thresh = eps*alpha*0.5

    if show:
        print('alpha = %10e\tthresh = %10e' % (alpha, thresh))
        print('-----------------------------------------------------------\n')
        head1 = '   Itn       x[0]        r2norm     r12norm     xupdate'
        print(head1)

    # initialize model and cost function
    xinv = np.zeros(Op.shape[1], dtype=Op.dtype)
    zinv = xinv.copy()
    t = 1
    if returninfo:
        cost = np.zeros(niter+1)

    # iterate
    for iiter in range(niter):
        xinvold = xinv.copy()

        # compute residual
        resz = data - Op.matvec(zinv)

        # compute gradient
        grad = alpha*Op.rmatvec(resz)

        # update inverted model
        xinv_unthesh = zinv + grad
        xinv = _softthreshold(xinv_unthesh, thresh)

        # update auxiliary coefficients
        told = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        zinv = xinv + ((told - 1.) / t) * (xinv - xinvold)

        # model update
        xupdate = np.linalg.norm(xinv - xinvold)

        if returninfo or show:
            costdata = 0.5*np.linalg.norm(data - Op.matvec(xinv))**2
            costreg = eps*np.linalg.norm(xinv, ord=1)
        if returninfo:
            cost[iiter] = costdata + costreg

        if show:
            if iiter < 10 or niter-iiter < 10 or iiter % 10 == 0:
                msg = '%6g  %12.5e  %10.3e   %9.3e  %10.3e' % \
                      (iiter+1, xinv[0], costdata, costdata+costreg, xupdate)
                print(msg)

        # check tolerance
        if xupdate < tol:
            niter = iiter
            break

    # get values pre-threshold  at locations where xinv is different from zero
    #xinv = np.where(xinv != 0, xinv_unthesh, xinv)
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (niter, time.time() - tstart))
        print('---------------------------------------------------------\n')
    if returninfo:
        return xinv, niter, cost[:niter]
    else:
        return xinv, niter


def SPGL1(Op, data, SOp=None, tau=0, sigma=0, x0=None, **kwargs_spgl1):
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
    data : :obj:`numpy.ndarray`
        Data
    SOp : :obj:`pylops.LinearOperator`
        Sparsyfing transform
    tau : :obj:`float`
        Non-negative LASSO scalar. If different from ``0``,
        SPGL1 will solve LASSO problem
    sigma : :obj:`list`
        BPDN scalar. If different from ``0``,
        SPGL1 will solve BPDN problem
    x0 : :obj:`numpy.ndarray`
        Initial guess
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

        ``tau``, final value of tau (see sigma above)

        ``rnorm``, two-norm of the optimal residual

        ``rgap``, relative duality gap (an optimality measure)

        ``gnorm``, Lagrange multiplier of (LASSO)

        ``stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4``: found a LASSO solution,
           ``5``: error, too many iterations,
           ``6``: error, linesearch failed,
           ``7``: error, found suboptimal BP solution,
           ``8``: error, too many matrix-vector products.

        ``niters``, number of iterations

        ``nProdA``, number of multiplications with A

        ``nProdAt``, number of multiplications with A'

        ``n_newton``, number of Newton steps

        ``time_project``, projection time (seconds)

        ``time_matprod``, matrix-vector multiplications time (seconds)

        ``time_total``, total solution time (seconds)

        ``niters_lsqr``, number of lsqr iterations (if ``subspace_min=True``)

        ``xnorm1``, L1-norm model solution history through iterations

        ``rnorm2``, L2-norm residual history through iterations

        ``lambdaa``, Lagrange multiplier history through iterations

    Raises
    ------
    ModuleNotFoundError
        If the ``spgl1`` library is not installed

    Notes
    -----
    Solve different variations of sparsity-promoting inverse problem by
    imposing sparsity in the retrieved model [1]_.

    The first problem is called *basis pursuit denoise (BPDN)* and
    its cost function is

        .. math::
            ||\mathbf{x}||_1 \quad  subj. to \quad
            ||\mathbf{Op}\mathbf{S}^H\mathbf{x}-\mathbf{b}||_2 <= \sigma,

    while the second problem is the *l1-regularized least-squares or LASSO*
    problem and its cost function is

        .. math::
            ||\mathbf{Op}\mathbf{S}^H\mathbf{x}-\mathbf{b}||_2 \quad  subj.
            to \quad  ||\mathbf{x}||_1  <= \tau

    .. [1] van den Berg E., Friedlander M.P., "Probing the Pareto frontier
       for basis pursuit solutions", SIAM J. on Scientific Computing,
       vol. 31(2), pp. 890-912. 2008.

    """
    if spgl1 is None:
        raise ModuleNotFoundError(spgl1_message)
    pinv, _, _, info = \
        spgl1(Op if SOp is None else Op*SOp.H, data,
              tau=tau, sigma=sigma, x0=x0, **kwargs_spgl1)

    xinv = pinv.copy() if SOp is None else SOp.H * pinv
    return xinv, pinv, info


def SplitBregman(Op, RegsL1, data, niter_outer=3, niter_inner=5, RegsL2=None,
                 dataregsL2=None, mu=1., epsRL1s=None, epsRL2s=None,
                 tol=1e-10, tau=1., x0=None, restart=False,
                 show=False, **kwargs_lsqr):
    r"""Split Bregman for mixed L2-L1 norms.

    Solve an unconstrained system of equations with mixed L2-L1 regularization
    terms given the operator ``Op``, a list of L1 regularization terms
    ``RegsL1``, and an optional list of L2 regularization terms ``RegsL2``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    RegsL1 : :obj:`list`
        L1 regularization operators
    data : :obj:`numpy.ndarray`
        Data
    niter_outer : :obj:`int`
        Number of iterations of outer loop
    niter_inner : :obj:`int`
        Number of iterations of inner loop
    RegsL2 : :obj:`list`
        Additional L2 regularization operators
        (if ``None``, L2 regularization is not added to the problem)
    dataregsL2 : :obj:`list`, optional
        L2 Regularization data (must have the same number of elements
        of ``RegsL2`` or equal to ``None`` to use a zero data for every
        regularization operator in ``RegsL2``)
    mu : :obj:`float`, optional
         Data term damping
    epsRL1s : :obj:`list`
         L1 Regularization dampings (must have the same number of elements
         as ``RegsL1``)
    epsRL2s : :obj:`list`
         L2 Regularization dampings (must have the same number of elements
         as ``RegsL2``)
    tol : :obj:`float`, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    tau : :obj:`float`, optional
        Scaling factor in the Bregman update (must be close to 1)
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    restart : :obj:`bool`, optional
        The unconstrained inverse problem in inner loop is initialized with
        the initial guess (``True``) or with the last estimate (``False``)
    show : :obj:`bool`, optional
        Display iterations log
    **kwargs_lsqr
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    itn_out : :obj:`int`
        Iteration number of outer loop upon termination

    Notes
    -----
    Solve the following system of unconstrained, regularized equations
    given the operator :math:`\mathbf{Op}` and a set of mixed norm (L2 and L1)
    regularization terms :math:`\mathbf{R_{L2,i}}` and
    :math:`\mathbf{R_{L1,i}}`, respectively:

    .. math::
        J = \mu/2 ||\textbf{d} - \textbf{Op} \textbf{x} |||_2 +
        \sum_i \epsilon_{{R}_{L2,i}} ||\mathbf{d_{{R}_{L2,i}}} -
        \mathbf{R_{L2,i}} \textbf{x} |||_2 +
        \sum_i || \mathbf{R_{L1,i}} \textbf{x} |||_1

    where :math:`\mu` and :math:`\epsilon_{{R}_{L2,i}}` are the damping factors
    used to weight the different terms of the cost function.

    The generalized Split Bergman algorithm is used to solve such cost
    function: the algorithm is composed of a sequence of unconstrained
    inverse problems and Bregman updates. Note that the L1 terms are not
    weighted in the  original cost function but are first converted into
    constraints and then re-inserted in the cost function with Lagrange
    multipliers :math:`\epsilon_{{R}_{L1,i}}`, which effectively act as
    damping factors for those terms. See [1]_ for detailed derivation.

    The :py:func:`scipy.sparse.linalg.lsqr` solver and a fast shrinkage
    algorithm are used within the inner loop to solve the unconstrained
    inverse problem, and the same procedure is repeated ``niter_outer`` times
    until convergence.

    .. [1] Goldstein T. and Osher S., "The Split Bregman Method for
       L1-Regularized Problems", SIAM J. on Scientific Computing, vol. 2(2),
       pp. 323-343. 2008.

    """
    if show:
        tstart = time.time()
        print('Split-Bregman optimization\n'
              '---------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'niter_outer = %3d     niter_inner = %3d   tol = %2.2e\n'
              'mu = %2.2e         epsL1 = %s\t  epsL2 = %s     '
              % (Op.shape[0], Op.shape[1],
                 niter_outer, niter_inner, tol,
                 mu, str(epsRL1s), str(epsRL2s)))
        print('---------------------------------------------------------\n')
        head1 = '   Itn          x[0]           r2norm          r12norm'
        print(head1)

    # L1 regularizations
    nregsL1 = len(RegsL1)
    b = [np.zeros(RegL1.shape[0]) for RegL1 in RegsL1]
    d = b.copy()

    # L2 regularizations
    nregsL2 = 0 if RegsL2 is None else len(RegsL2)
    if nregsL2 > 0:
        Regs = RegsL2 + RegsL1
        if dataregsL2 is None:
            dataregsL2 = [np.zeros(Op.shape[1])] * nregsL2
    else:
        Regs = RegsL1
        dataregsL2 = []

    # Rescale dampings
    epsRs = [np.sqrt(epsRL2s[ireg] / 2) / np.sqrt(mu / 2) for ireg in
             range(nregsL2)] + \
            [np.sqrt(epsRL1s[ireg] / 2) / np.sqrt(mu / 2) for ireg in
             range(nregsL1)]
    xinv = np.zeros_like(np.zeros(Op.shape[1])) if x0 is None else x0
    xold = np.inf * np.ones_like(np.zeros(Op.shape[1]))

    itn_out = 0
    while np.linalg.norm(xinv - xold) > tol and itn_out < niter_outer:
        xold = xinv
        for _ in range(niter_inner):
            # Regularized problem
            dataregs = \
                dataregsL2 + [d[ireg] - b[ireg] for ireg in range(nregsL1)]
            xinv = RegularizedInversion(Op, Regs, data,
                                        dataregs=dataregs,
                                        epsRs=epsRs,
                                        x0=x0 if restart else xinv,
                                        **kwargs_lsqr)
            # Shrinkage
            d = [_shrinkage(RegsL1[ireg] * xinv + b[ireg], epsRL1s[ireg])
                 for ireg in range(nregsL1)]
        # Bregman update
        b = [b[ireg] + tau * (RegsL1[ireg] * xinv - d[ireg]) for ireg in
             range(nregsL1)]
        itn_out += 1

        if show:
            costdata = mu/2. * np.linalg.norm(data - Op.matvec(xinv)) ** 2
            costregL2 = 0 if RegsL2 is None else \
                [epsRL2 * np.linalg.norm(dataregL2 - RegL2.matvec(xinv)) ** 2
                 for epsRL2, RegL2, dataregL2 in zip(epsRL2s, RegsL2, dataregsL2)]
            costregL1 = [np.linalg.norm(RegL1.matvec(xinv), ord=1)
                         for epsRL1, RegL1 in zip(epsRL1s, RegsL1)]
            cost = costdata + np.sum(np.array(costregL2)) + \
                   np.sum(np.array(costregL1))
            msg = '%6g  %12.5e       %10.3e        %9.3e' % \
                  (np.abs(itn_out), xinv[0], costdata, cost)
            print(msg)

    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (itn_out, time.time() - tstart))
        print('---------------------------------------------------------\n')
    return xinv, itn_out
