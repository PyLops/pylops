import logging
import time

import numpy as np
from scipy.sparse.linalg import lsqr

from pylops import LinearOperator
from pylops.basicoperators import Diagonal, Identity
from pylops.optimization.eigs import power_iteration
from pylops.optimization.leastsquares import (
    NormalEquationsInversion,
    RegularizedInversion,
)
from pylops.optimization.solver import cgls
from pylops.utils.backend import get_array_module, get_module_name, to_numpy

try:
    from spgl1 import spgl1
except ModuleNotFoundError:
    spgl1 = None
    spgl1_message = "Spgl1 not installed. " 'Run "pip install spgl1".'
except Exception as e:
    spgl1 = None
    spgl1_message = f"Failed to import spgl1 (error:{e})."


def _hardthreshold(x, thresh):
    r"""Hard thresholding.

    Applies hard thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_0`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
       operator of the ℓp norm with 0 < p < 1”,
       IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    x1 = x.copy()
    x1[np.abs(x) <= np.sqrt(2 * thresh)] = 0
    return x1


def _softthreshold(x, thresh):
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_1`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
       operator of the ℓp norm with 0 < p < 1”,
       IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.0) * np.exp(1j * np.angle(x))
    else:
        x1 = np.maximum(np.abs(x) - thresh, 0.0) * np.sign(x)
    return x1


def _halfthreshold(x, thresh):
    r"""Half thresholding.

    Applies half thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_{1/2}^{1/2}`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., “Computing the proximity
       operator of the ℓp norm with 0 < p < 1”,
       IET Signal Processing, vol. 10. 2016.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

        .. warning::
            Since version 1.17.0 does not produce ``np.nan`` on bad input.

    """
    arg = np.ones_like(x)
    arg[x != 0] = (thresh / 8.0) * (np.abs(x[x != 0]) / 3.0) ** (-1.5)
    arg = np.clip(arg, -1, 1)
    phi = 2.0 / 3.0 * np.arccos(arg)
    x1 = 2.0 / 3.0 * x * (1 + np.cos(2.0 * np.pi / 3.0 - phi))
    # x1[np.abs(x) <= 1.5 * thresh ** (2. / 3.)] = 0
    x1[np.abs(x) <= (54 ** (1.0 / 3.0) / 4.0) * thresh ** (2.0 / 3.0)] = 0
    return x1


def _hardthreshold_percentile(x, perc):
    r"""Percentile Hard thresholding.

    Applies hard thresholding to vector ``x`` using a percentile to define
    the amount of values in the input vector to be preserved as shown in [1]_.

    .. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
       data reconstruction using a percentile-half-thresholding algorithm”,
       Journal of Geophysics and Engineering, vol. 11. 2014.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    thresh = np.percentile(np.abs(x), perc)
    return _hardthreshold(x, 0.5 * thresh ** 2)


def _softthreshold_percentile(x, perc):
    r"""Percentile Soft thresholding.

    Applies soft thresholding to vector ``x`` using a percentile to define
    the amount of values in the input vector to be preserved as shown in [1]_.

    .. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
       data reconstruction using a percentile-half-thresholding algorithm”,
       Journal of Geophysics and Engineering, vol. 11. 2014.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    perc : :obj:`float`
        Percentile

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Tresholded vector

    """
    thresh = np.percentile(np.abs(x), perc)
    return _softthreshold(x, thresh)


def _halfthreshold_percentile(x, perc):
    r"""Percentile Half thresholding.

    Applies half thresholding to vector ``x`` using a percentile to define
    the amount of values in the input vector to be preserved as shown in [1]_.

    .. [1] Xu, Z., Xiangyu, C., Xu, F. and Zhang, H., “L1/2 Regularization: A
       Thresholding Representation Theory and a Fast Solver”, IEEE Transactions
       on Neural Networks and Learning Systems, vol. 23. 2012.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    perc : :obj:`float`
        Percentile

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Tresholded vector

    """
    thresh = np.percentile(np.abs(x), perc)
    # return _halfthreshold(x, (2. / 3. * thresh) ** (1.5))
    return _halfthreshold(x, (4.0 / 54 ** (1.0 / 3.0) * thresh) ** 1.5)


def _IRLS_data(
    Op,
    data,
    nouter,
    threshR=False,
    epsR=1e-10,
    epsI=1e-10,
    x0=None,
    tolIRLS=1e-10,
    returnhistory=False,
    **kwargs_solver,
):
    r"""Iteratively reweighted least squares with L1 data term"""
    ncp = get_array_module(data)

    if x0 is not None:
        data = data - Op * x0
    if returnhistory:
        xinv_hist = ncp.zeros((nouter + 1, int(Op.shape[1])))
        rw_hist = ncp.zeros((nouter + 1, int(Op.shape[0])))

    # first iteration (unweighted least-squares)
    xinv = NormalEquationsInversion(
        Op, None, data, epsI=epsI, returninfo=False, **kwargs_solver
    )
    r = data - Op * xinv
    if returnhistory:
        xinv_hist[0] = xinv
    for iiter in range(nouter):
        # other iterations (weighted least-squares)
        xinvold = xinv.copy()
        if threshR:
            rw = 1.0 / ncp.maximum(ncp.abs(r), epsR)
        else:
            rw = 1.0 / (ncp.abs(r) + epsR)
        rw = rw / rw.max()
        R = Diagonal(rw)
        xinv = NormalEquationsInversion(
            Op, [], data, Weight=R, epsI=epsI, returninfo=False, **kwargs_solver
        )
        r = data - Op * xinv
        # save history
        if returnhistory:
            rw_hist[iiter] = rw
            xinv_hist[iiter + 1] = xinv
        # check tolerance
        if ncp.linalg.norm(xinv - xinvold) < tolIRLS:
            nouter = iiter
            break

    # adding initial guess
    if x0 is not None:
        xinv = x0 + xinv
        if returnhistory:
            xinv_hist = x0 + xinv_hist

    if returnhistory:
        return xinv, nouter, xinv_hist[: nouter + 1], rw_hist[: nouter + 1]
    else:
        return xinv, nouter


def _IRLS_model(
    Op,
    data,
    nouter,
    threshR=False,
    epsR=1e-10,
    epsI=1e-10,
    x0=None,
    tolIRLS=1e-10,
    returnhistory=False,
    **kwargs_solver,
):
    r"""Iteratively reweighted least squares with L1 model term"""
    ncp = get_array_module(data)

    if x0 is not None:
        data = data - Op * x0
    if returnhistory:
        xinv_hist = ncp.zeros((nouter + 1, int(Op.shape[1])))
        rw_hist = ncp.zeros((nouter + 1, int(Op.shape[1])))

    Iop = Identity(data.size, dtype=data.dtype)
    # first iteration (unweighted least-squares)
    if ncp == np:
        xinv = Op.H @ lsqr(Op @ Op.H + (epsI ** 2) * Iop, data, **kwargs_solver)[0]
    else:
        xinv = Op.H @ cgls(
            Op @ Op.H + (epsI ** 2) * Iop,
            data,
            ncp.zeros(int(Op.shape[0]), dtype=Op.dtype),
            **kwargs_solver,
        )[0]
    if returnhistory:
        xinv_hist[0] = xinv
    for iiter in range(nouter):
        # other iterations (weighted least-squares)
        xinvold = xinv.copy()
        rw = np.abs(xinv)
        rw = rw / rw.max()
        R = Diagonal(rw, dtype=rw.dtype)
        if ncp == np:
            xinv = (
                R
                @ Op.H
                @ lsqr(Op @ R @ Op.H + epsI ** 2 * Iop, data, **kwargs_solver)[0]
            )
        else:
            xinv = (
                R
                @ Op.H
                @ cgls(
                    Op @ R @ Op.H + epsI ** 2 * Iop,
                    data,
                    ncp.zeros(int(Op.shape[0]), dtype=Op.dtype),
                    **kwargs_solver,
                )[0]
            )
        # save history
        if returnhistory:
            rw_hist[iiter] = rw
            xinv_hist[iiter + 1] = xinv
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
        return xinv, nouter, xinv_hist[: nouter + 1], rw_hist[: nouter + 1]
    else:
        return xinv, nouter


def IRLS(
    Op,
    data,
    nouter,
    threshR=False,
    epsR=1e-10,
    epsI=1e-10,
    x0=None,
    tolIRLS=1e-10,
    returnhistory=False,
    kind="data",
    **kwargs_solver,
):
    r"""Iteratively reweighted least squares.

    Solve an optimization problem with :math:`L^1` cost function (data IRLS)
    or :math:`L^1` regularization term (model IRLS) given the operator ``Op``
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
    kind : :obj:`str`, optional
        Kind of solver (``data`` or ``model``)
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
    xinv_hist : :obj:`numpy.ndarray`, optional
        History of inverted model
    rw_hist : :obj:`numpy.ndarray`, optional
        History of weights

    Notes
    -----
    *Data IRLS* solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = \|\mathbf{d} - \mathbf{Op}\,\mathbf{x}\|_1

    by a set of outer iterations which require to repeatedly solve a
    weighted least squares problem of the form:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \mathbf{x}^{(i+1)} = \argmin_\mathbf{x} \|\mathbf{d} -
        \mathbf{Op}\,\mathbf{x}\|_{2, \mathbf{R}^{(i)}}^2 +
        \epsilon_\mathbf{I}^2 \|\mathbf{x}\|_2^2

    where :math:`\mathbf{R}^{(i)}` is a diagonal weight matrix
    whose diagonal elements at iteration :math:`i` are equal to the absolute
    inverses of the residual vector :math:`\mathbf{r}^{(i)} =
    \mathbf{y} - \mathbf{Op}\,\mathbf{x}^{(i)}` at iteration :math:`i`.
    More specifically the :math:`j`-th element of the diagonal of
    :math:`\mathbf{R}^{(i)}` is

    .. math::
        R^{(i)}_{j,j} = \frac{1}{\left| r^{(i)}_j \right| + \epsilon_\mathbf{R}}

    or

    .. math::
        R^{(i)}_{j,j} = \frac{1}{\max\{\left|r^{(i)}_j\right|, \epsilon_\mathbf{R}\}}

    depending on the choice ``threshR``. In either case,
    :math:`\epsilon_\mathbf{R}` is the user-defined stabilization/thresholding
    factor [1]_.

    Similarly *model IRLS* solves the following optimization problem for the
    operator :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = \|\mathbf{x}\|_1 \quad \text{subject to} \quad
        \mathbf{d} = \mathbf{Op}\,\mathbf{x}

    by a set of outer iterations which require to repeatedly solve a
    weighted least squares problem of the form [2]_:

    .. math::
        \mathbf{x}^{(i+1)} = \operatorname*{arg\,min}_\mathbf{x}
        \|\mathbf{x}\|_{2, \mathbf{R}^{(i)}}^2 \quad \text{subject to} \quad
        \mathbf{d} = \mathbf{Op}\,\mathbf{x}

    where :math:`\mathbf{R}^{(i)}` is a diagonal weight matrix
    whose diagonal elements at iteration :math:`i` are equal to the absolutes
    of the model vector :math:`\mathbf{x}^{(i)}` at iteration
    :math:`i`. More specifically the :math:`j`-th element of the diagonal of
    :math:`\mathbf{R}^{(i)}` is

    .. math::
        R^{(i)}_{j,j} = \left|x^{(i)}_j\right|.

    .. [1] https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
    .. [2] Chartrand, R., and Yin, W. "Iteratively reweighted algorithms for
       compressive sensing", IEEE. 2008.

    """
    if kind == "data":
        solver = _IRLS_data
    elif kind == "model":
        solver = _IRLS_model
    else:
        raise NotImplementedError("kind must be data or model")
    return solver(
        Op,
        data,
        nouter,
        threshR=threshR,
        epsR=epsR,
        epsI=epsI,
        x0=x0,
        tolIRLS=tolIRLS,
        returnhistory=returnhistory,
        **kwargs_solver,
    )


def OMP(
    Op,
    data,
    niter_outer=10,
    niter_inner=40,
    sigma=1e-4,
    normalizecols=False,
    show=False,
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
    data : :obj:`numpy.ndarray`
        Data
    niter_outer : :obj:`int`, optional
        Number of iterations of outer loop
    niter_inner : :obj:`int`, optional
        Number of iterations of inner loop. By choosing ``niter_inner=0``, the
        Matching Pursuit (MP) algorithm is implemented.
    sigma : :obj:`list`
        Maximum :math:`L^2` norm of residual. When smaller stop iterations.
    normalizecols : :obj:`list`, optional
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
    ISTA: Iterative Shrinkage-Thresholding Algorithm (ISTA).
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
            \|\mathbf{x}\|_0 \quad  \text{subject to} \quad
            \|\mathbf{Op}\,\mathbf{x}-\mathbf{b}\|_2^2 \leq \sigma^2,

    using Orthogonal Matching Pursuit (OMP). This is a very
    simple iterative algorithm which applies the following step:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \DeclareMathOperator*{\argmax}{arg\,max}
        \Lambda_k = \Lambda_{k-1} \cup \left\{\argmax_j
        \left|\mathbf{Op}_j^H\,\mathbf{r}_k\right| \right\} \\
        \mathbf{x}_k = \argmin_{\mathbf{x}}
        \left\|\mathbf{Op}_{\Lambda_k}\,\mathbf{x} - \mathbf{b}\right\|_2^2

    Note that by choosing ``niter_inner=0`` the basic Matching Pursuit (MP)
    algorithm is implemented instead. In other words, instead of solving an
    optimization at each iteration to find the best :math:`\mathbf{x}` for the
    currently selected basis functions, the vector :math:`\mathbf{x}` is just
    updated at the new basis function by taking directly the value from
    the inner product :math:`\mathbf{Op}_j^H\,\mathbf{r}_k`.

    In this case it is highly reccomended to provide a normalized basis
    function. If different basis have different norms, the solver is likely
    to diverge. Similar observations apply to OMP, even though mild unbalancing
    between the basis is generally properly handled.

    """
    ncp = get_array_module(data)

    Op = LinearOperator(Op)
    if show:
        tstart = time.time()
        algname = "OMP optimization\n" if niter_inner > 0 else "MP optimization\n"
        print(
            algname
            + "-----------------------------------------------------------------\n"
            f"The Operator Op has {Op.shape[0]} rows and {Op.shape[1]} cols\n"
            f"sigma = {sigma:.2e}\tniter_outer = {niter_outer}\tniter_inner = {niter_inner}\n"
            f"normalization={normalizecols}"
        )
    # find normalization factor for each column
    if normalizecols:
        ncols = Op.shape[1]
        norms = ncp.zeros(ncols)
        for icol in range(ncols):
            unit = ncp.zeros(ncols, dtype=Op.dtype)
            unit[icol] = 1
            norms[icol] = np.linalg.norm(Op.matvec(unit))
    if show:
        print("-----------------------------------------------------------------")
        head1 = "    Itn           r2norm"
        print(head1)

    if niter_inner == 0:
        x = []
    cols = []
    res = data.copy()
    cost = ncp.zeros(niter_outer + 1)
    cost[0] = np.linalg.norm(data)
    iiter = 0
    while iiter < niter_outer and cost[iiter] > sigma:
        # compute inner products
        cres = Op.rmatvec(res)
        cres_abs = np.abs(cres)
        if normalizecols:
            cres_abs = cres_abs / norms
        # choose column with max cres
        cres_max = np.max(cres_abs)
        imax = np.argwhere(cres_abs == cres_max).ravel()
        nimax = len(imax)
        if nimax > 0:
            imax = imax[np.random.permutation(nimax)[0]]
        else:
            imax = imax[0]
        # update active set
        if imax not in cols:
            addnew = True
            cols.append(int(imax))
        else:
            addnew = False
            imax_in_cols = cols.index(imax)

        # estimate model for current set of columns
        if niter_inner == 0:
            # MP update
            Opcol = Op.apply_columns(
                [
                    int(imax),
                ]
            )
            res -= Opcol.matvec(cres[imax] * ncp.ones(1))
            if addnew:
                x.append(cres[imax])
            else:
                x[imax_in_cols] += cres[imax]
        else:
            # OMP update
            Opcol = Op.apply_columns(cols)
            if ncp == np:
                x = lsqr(Opcol, data, iter_lim=niter_inner)[0]
            else:
                x = cgls(
                    Opcol,
                    data,
                    ncp.zeros(int(Opcol.shape[1]), dtype=Opcol.dtype),
                    niter=niter_inner,
                )[0]
            res = data - Opcol.matvec(x)
        iiter += 1
        cost[iiter] = np.linalg.norm(res)
        if show:
            if iiter < 10 or niter_outer - iiter < 10 or iiter % 10 == 0:
                msg = f"{iiter + 1:6g}        {cost[iiter]:12.5e}"
                print(msg)
    xinv = ncp.zeros(int(Op.shape[1]), dtype=Op.dtype)
    xinv[cols] = ncp.array(x)
    if show:
        print(
            f"\nIterations = {iiter}        Total time (s) = {time.time() - tstart:.2f}"
        )
        print("-----------------------------------------------------------------\n")
    return xinv, iiter, cost


def ISTA(
    Op,
    data,
    niter,
    eps=0.1,
    alpha=None,
    eigsiter=None,
    eigstol=0,
    tol=1e-10,
    monitorres=False,
    returninfo=False,
    show=False,
    threshkind="soft",
    perc=None,
    callback=None,
    decay=None,
    SOp=None,
    x0=None,
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
    data : :obj:`numpy.ndarray`
        Data
    niter : :obj:`int`
        Number of iterations
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
    monitorres : :obj:`bool`, optional
        Monitor that residual is decreasing
    returninfo : :obj:`bool`, optional
        Return info of CG solver
    show : :obj:`bool`, optional
        Display iterations log
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half', 'hard-percentile',
        'soft-percentile', or 'half-percentile' - 'soft' used as default)
    perc : :obj:`float`, optional
        Percentile, as percentage of values to be kept by thresholding (to be
        provided when thresholding is soft-percentile or half-percentile)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    SOp : :obj:`pylops.LinearOperator`, optional
        Regularization operator (use when solving the analysis problem)
    x0: :obj:`numpy.ndarray`, optional
        Initial guess

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
    Solves the following synthesis problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = \|\mathbf{d} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{x}\|_p

    or the analysis problem:

    .. math::
        J = \|\mathbf{d} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{SOp}^H\,\mathbf{x}\|_p

    if ``SOp`` is provided. Note that in the first case, ``SOp`` should be
    assimilated in the modelling operator (i.e., ``Op=GOp * SOp``).

    The Iterative Shrinkage-Thresholding Algorithms (ISTA) [1]_ is used, where
    :math:`p=0, 0.5, 1`. This is a very simple iterative algorithm which
    applies the following step:

    .. math::
        \mathbf{x}^{(i+1)} = T_{(\epsilon \alpha /2, p)} \left(\mathbf{x}^{(i)} +
        \alpha\,\mathbf{Op}^H \left(\mathbf{d} - \mathbf{Op}\,\mathbf{x}^{(i)}\right)\right)

    or

    .. math::
        \mathbf{x}^{(i+1)} = \mathbf{SOp}\,\left\{T_{(\epsilon \alpha /2, p)}
        \mathbf{SOp}^H\,\left(\mathbf{x}^{(i)} + \alpha\,\mathbf{Op}^H \left(\mathbf{d} -
        \mathbf{Op} \,\mathbf{x}^{(i)}\right)\right)\right\}

    where :math:`\epsilon \alpha /2` is the threshold and :math:`T_{(\tau, p)}`
    is the thresholding rule. The most common variant of ISTA uses the
    so-called soft-thresholding rule :math:`T(\tau, p=1)`. Alternatively an
    hard-thresholding rule is used in the case of :math:`p=0` or a half-thresholding
    rule is used in the case of :math:`p=1/2`. Finally, percentile bases thresholds
    are also implemented: the damping factor is not used anymore an the
    threshold changes at every iteration based on the computed percentile.

    .. [1] Daubechies, I., Defrise, M., and De Mol, C., “An iterative
       thresholding algorithm for linear inverse problems with a sparsity
       constraint”, Communications on pure and applied mathematics, vol. 57,
       pp. 1413-1457. 2004.

    """
    if not threshkind in [
        "hard",
        "soft",
        "half",
        "hard-percentile",
        "soft-percentile",
        "half-percentile",
    ]:
        raise NotImplementedError(
            "threshkind should be hard, soft, half,"
            "hard-percentile, soft-percentile, "
            "or half-percentile"
        )
    if (
        threshkind in ["hard-percentile", "soft-percentile", "half-percentile"]
        and perc is None
    ):
        raise ValueError(
            "Provide a percentile when choosing hard-percentile,"
            "soft-percentile, or half-percentile thresholding"
        )

    # choose thresholding function
    if threshkind == "soft":
        threshf = _softthreshold
    elif threshkind == "hard":
        threshf = _hardthreshold
    elif threshkind == "half":
        threshf = _halfthreshold
    elif threshkind == "hard-percentile":
        threshf = _hardthreshold_percentile
    elif threshkind == "soft-percentile":
        threshf = _softthreshold_percentile
    else:
        threshf = _halfthreshold_percentile

    # identify backend to use
    ncp = get_array_module(data)

    # prepare decay (if not passed)
    if perc is None and decay is None:
        decay = ncp.ones(niter)

    if show:
        tstart = time.time()
        print(
            f"ISTA optimization ({threshkind} thresholding)\n"
            "-----------------------------------------------------------\n"
            f"The Operator Op has {Op.shape[0]} rows and {Op.shape[1]} cols\n"
            f"eps = {eps:10e}\ttol = {tol:10e}\tniter = {niter}"
        )
    # step size
    if alpha is None:
        if not isinstance(Op, LinearOperator):
            Op = LinearOperator(Op, explicit=False)
        # compute largest eigenvalues of Op^H * Op
        Op1 = LinearOperator(Op.H * Op, explicit=False)
        if get_module_name(ncp) == "numpy":
            maxeig = np.abs(
                Op1.eigs(
                    neigs=1,
                    symmetric=True,
                    niter=eigsiter,
                    **dict(tol=eigstol, which="LM"),
                )[0]
            )
        else:
            maxeig = np.abs(
                power_iteration(
                    Op1, niter=eigsiter, tol=eigstol, dtype=Op1.dtype, backend="cupy"
                )[0]
            )
        alpha = 1.0 / maxeig

    # define threshold
    thresh = eps * alpha * 0.5

    if show:
        if perc is None:
            print(f"alpha = {alpha:10e}\tthresh = {thresh:10e}")
        else:
            print(f"alpha = {alpha:10e}\tperc = {perc:.1f}")
        print("-----------------------------------------------------------\n")
        head1 = "   Itn       x[0]        r2norm     r12norm     xupdate"
        print(head1)

    # initialize model and cost function
    if x0 is None:
        if data.ndim == 1:
            xinv = ncp.zeros(int(Op.shape[1]), dtype=Op.dtype)
        else:
            xinv = ncp.zeros((int(Op.shape[1]), data.shape[1]), dtype=Op.dtype)
    else:
        if data.ndim != x0.ndim:
            # error for wrong dimensions
            raise ValueError("Number of columns of x0 and data are not the same")
        elif x0.shape[0] != Op.shape[1]:
            # error for wrong dimensions
            raise ValueError("Operator and input vector have different dimensions")
        else:
            xinv = x0.copy()

    if monitorres:
        normresold = np.inf
    if returninfo:
        cost = np.zeros(niter + 1)

    # iterate
    for iiter in range(niter):
        xinvold = xinv.copy()

        # compute residual
        res = data - Op @ xinv
        if monitorres:
            normres = np.linalg.norm(res)
            if normres > normresold:
                raise ValueError(
                    f"ISTA stopped at iteration {iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                normresold = normres

        # compute gradient
        grad = alpha * Op.H @ res

        # update inverted model
        xinv_unthesh = xinv + grad
        if SOp is not None:
            xinv_unthesh = SOp.H @ xinv_unthesh
        if perc is None:
            xinv = threshf(xinv_unthesh, decay[iiter] * thresh)
        else:
            xinv = threshf(xinv_unthesh, 100 - perc)
        if SOp is not None:
            xinv = SOp @ xinv

        # model update
        xupdate = np.linalg.norm(xinv - xinvold)

        if returninfo or show:
            costdata = 0.5 * np.linalg.norm(res) ** 2
            costreg = eps * np.linalg.norm(xinv, ord=1)
        if returninfo:
            cost[iiter] = costdata + costreg

        # run callback
        if callback is not None:
            callback(xinv)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                msg = (
                    f"{iiter + 1:6g}  {to_numpy(xinv[:2])[0]:12.5e}  "
                    f"{costdata:10.3e}   {costdata + costreg:9.3e}  {xupdate:10.3e}"
                )
                print(msg)

        # check tolerance
        if xupdate < tol:
            logging.warning("update smaller that tolerance for " "iteration %d", iiter)
            niter = iiter
            break

    # get values pre-threshold at locations where xinv is different from zero
    # xinv = np.where(xinv != 0, xinv_unthesh, xinv)

    if show:
        print(
            f"\nIterations = {niter}        Total time (s) = {time.time() - tstart:.2f}"
        )
        print("---------------------------------------------------------\n")
    if returninfo:
        return xinv, niter, cost[:niter]
    else:
        return xinv, niter


def FISTA(
    Op,
    data,
    niter,
    eps=0.1,
    alpha=None,
    eigsiter=None,
    eigstol=0,
    tol=1e-10,
    returninfo=False,
    show=False,
    threshkind="soft",
    perc=None,
    callback=None,
    decay=None,
    SOp=None,
    x0=None,
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
    data : :obj:`numpy.ndarray`
        Data
    niter : :obj:`int`
        Number of iterations
    eps : :obj:`float`, optional
        Sparsity damping
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
    returninfo : :obj:`bool`, optional
        Return info of FISTA solver
    show : :obj:`bool`, optional
        Display iterations log
    threshkind : :obj:`str`, optional
        Kind of thresholding ('hard', 'soft', 'half', 'soft-percentile', or
        'half-percentile' - 'soft' used as default)
    perc : :obj:`float`, optional
        Percentile, as percentage of values to be kept by thresholding (to be
        provided when thresholding is soft-percentile or half-percentile)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    decay : :obj:`numpy.ndarray`, optional
        Decay factor to be applied to thresholding during iterations
    SOp : :obj:`pylops.LinearOperator`, optional
        Regularization operator (use when solving the analysis problem)
    x0: :obj:`numpy.ndarray`, optional
        Initial guess

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
    Solves the following synthesis problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = \|\mathbf{d} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{x}\|_p

    or the analysis problem:

    .. math::
        J = \|\mathbf{d} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{SOp}^H\,\mathbf{x}\|_p

    if ``SOp`` is provided.

    The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) [1]_ is used,
    where :math:`p=0, 0.5, 1`. This is a modified version of ISTA solver with
    improved convergence properties and limited additional computational cost.
    Similarly to the ISTA solver, the choice of the thresholding algorithm to
    apply at every iteration is based on the choice of :math:`p`.

    .. [1] Beck, A., and Teboulle, M., “A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems”, SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    if not threshkind in [
        "hard",
        "soft",
        "half",
        "hard-percentile",
        "soft-percentile",
        "half-percentile",
    ]:
        raise NotImplementedError(
            "threshkind should be hard, soft, half,"
            "hard-percentile, soft-percentile, "
            "or half-percentile"
        )
    if (
        threshkind in ["hard-percentile", "soft-percentile", "half-percentile"]
        and perc is None
    ):
        raise ValueError(
            "Provide a percentile when choosing hard-percentile,"
            "soft-percentile, or half-percentile thresholding"
        )

    # choose thresholding function
    if threshkind == "soft":
        threshf = _softthreshold
    elif threshkind == "hard":
        threshf = _hardthreshold
    elif threshkind == "half":
        threshf = _halfthreshold
    elif threshkind == "hard-percentile":
        threshf = _hardthreshold_percentile
    elif threshkind == "soft-percentile":
        threshf = _softthreshold_percentile
    else:
        threshf = _halfthreshold_percentile

    # identify backend to use
    ncp = get_array_module(data)

    # prepare decay (if not passed)
    if perc is None and decay is None:
        decay = ncp.ones(niter)

    if show:
        tstart = time.time()
        print(
            f"FISTA optimization ({threshkind} thresholding)\n"
            "-----------------------------------------------------------\n"
            f"The Operator Op has {Op.shape[0]} rows and {Op.shape[1]} cols\n"
            f"eps = {eps:10e}\ttol = {tol:10e}\tniter = {niter}"
        )
    # step size
    if alpha is None:
        if not isinstance(Op, LinearOperator):
            Op = LinearOperator(Op, explicit=False)
        # compute largest eigenvalues of Op^H * Op
        Op1 = LinearOperator(Op.H * Op, explicit=False)
        if get_module_name(ncp) == "numpy":
            maxeig = np.abs(
                Op1.eigs(
                    neigs=1,
                    symmetric=True,
                    niter=eigsiter,
                    **dict(tol=eigstol, which="LM"),
                )
            )[0]
        else:
            maxeig = np.abs(
                power_iteration(
                    Op1, niter=eigsiter, tol=eigstol, dtype=Op1.dtype, backend="cupy"
                )[0]
            )
        alpha = 1.0 / maxeig

    # define threshold
    thresh = eps * alpha * 0.5

    if show:
        if perc is None:
            print(f"alpha = {alpha:10e}\tthresh = {thresh:10e}")
        else:
            print(f"alpha = {alpha:10e}\tperc = {perc:.1f}")
        print("-----------------------------------------------------------\n")
        head1 = "   Itn       x[0]        r2norm     r12norm     xupdate"
        print(head1)

    # initialize model and cost function
    if x0 is None:
        if data.ndim == 1:
            xinv = ncp.zeros(int(Op.shape[1]), dtype=Op.dtype)
        else:
            xinv = ncp.zeros((int(Op.shape[1]), data.shape[1]), dtype=Op.dtype)
    else:
        if data.ndim != x0.ndim:
            # error for wrong dimensions
            raise ValueError("Number of columns of x0 and data are not the same")
        elif x0.shape[0] != Op.shape[1]:
            # error for wrong dimensions
            raise ValueError("Operator and input vector have different dimensions")
        else:
            xinv = x0.copy()

    zinv = xinv.copy()
    t = 1
    if returninfo:
        cost = np.zeros(niter + 1)

    # iterate
    for iiter in range(niter):
        xinvold = xinv.copy()

        # compute residual
        resz = data - Op @ zinv

        # compute gradient
        grad = alpha * Op.H @ resz

        # update inverted model
        xinv_unthesh = zinv + grad
        if SOp is not None:
            xinv_unthesh = SOp.H @ xinv_unthesh
        if perc is None:
            xinv = threshf(xinv_unthesh, decay[iiter] * thresh)
        else:
            xinv = threshf(xinv_unthesh, 100 - perc)
        if SOp is not None:
            xinv = SOp @ xinv

        # update auxiliary coefficients
        told = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        zinv = xinv + ((told - 1.0) / t) * (xinv - xinvold)

        # model update
        xupdate = np.linalg.norm(xinv - xinvold)

        if returninfo or show:
            costdata = 0.5 * np.linalg.norm(data - Op @ xinv) ** 2
            costreg = eps * np.linalg.norm(xinv, ord=1)
        if returninfo:
            cost[iiter] = costdata + costreg

        # run callback
        if callback is not None:
            callback(xinv)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                msg = (
                    f"{iiter + 1:6g}  {to_numpy(xinv[:2])[0]:12.5e}  "
                    f"{costdata:10.3e}   {costdata + costreg:9.3e}  {xupdate:10.3e}"
                )
                print(msg)

        # check tolerance
        if xupdate < tol:
            niter = iiter
            break

    # get values pre-threshold  at locations where xinv is different from zero
    # xinv = np.where(xinv != 0, xinv_unthesh, xinv)

    if show:
        print(
            f"\nIterations = {niter}        Total time (s) = {time.time() - tstart:.2f}"
        )
        print("---------------------------------------------------------\n")
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
    Solve different variations of sparsity-promoting inverse problem by
    imposing sparsity in the retrieved model [1]_.

    The first problem is called *basis pursuit denoise (BPDN)* and
    its cost function is

        .. math::
            \|\mathbf{x}\|_1 \quad  \text{subject to} \quad
            \left\|\mathbf{Op}\,\mathbf{S}^H\mathbf{x}-\mathbf{b}\right\|_2^2
            \leq \sigma,

    while the second problem is the *ℓ₁-regularized least-squares or LASSO*
    problem and its cost function is

        .. math::
            \left\|\mathbf{Op}\,\mathbf{S}^H\mathbf{x}-\mathbf{b}\right\|_2^2
            \quad \text{subject to} \quad  \|\mathbf{x}\|_1  \leq \tau

    .. [1] van den Berg E., Friedlander M.P., "Probing the Pareto frontier
       for basis pursuit solutions", SIAM J. on Scientific Computing,
       vol. 31(2), pp. 890-912. 2008.

    """
    if spgl1 is None:
        raise ModuleNotFoundError(spgl1_message)
    pinv, _, _, info = spgl1(
        Op if SOp is None else Op * SOp.H,
        data,
        tau=tau,
        sigma=sigma,
        x0=x0,
        **kwargs_spgl1,
    )

    xinv = pinv.copy() if SOp is None else SOp.H * pinv
    return xinv, pinv, info


def SplitBregman(
    Op,
    RegsL1,
    data,
    niter_outer=3,
    niter_inner=5,
    RegsL2=None,
    dataregsL2=None,
    mu=1.0,
    epsRL1s=None,
    epsRL2s=None,
    tol=1e-10,
    tau=1.0,
    x0=None,
    restart=False,
    show=False,
    **kwargs_lsqr,
):
    r"""Split Bregman for mixed L2-L1 norms.

    Solve an unconstrained system of equations with mixed :math:`L^2` and :math:`L^1`
    regularization terms given the operator ``Op``, a list of :math:`L^1`
    regularization terms ``RegsL1``, and an optional list of :math:`L^2`
    regularization terms ``RegsL2``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    RegsL1 : :obj:`list`
        :math:`L^1` regularization operators
    data : :obj:`numpy.ndarray`
        Data
    niter_outer : :obj:`int`
        Number of iterations of outer loop
    niter_inner : :obj:`int`
        Number of iterations of inner loop of first step of the Split Bregman
        algorithm. A small number of iterations is generally sufficient and
        for many applications optimal efficiency is obtained when only one
        iteration is performed.
    RegsL2 : :obj:`list`
        Additional :math:`L^2` regularization operators
        (if ``None``, :math:`L^2` regularization is not added to the problem)
    dataregsL2 : :obj:`list`, optional
        :math:`L^2` Regularization data (must have the same number of elements
        of ``RegsL2`` or equal to ``None`` to use a zero data for every
        regularization operator in ``RegsL2``)
    mu : :obj:`float`, optional
         Data term damping
    epsRL1s : :obj:`list`
         :math:`L^1` Regularization dampings (must have the same number of elements
         as ``RegsL1``)
    epsRL2s : :obj:`list`
         :math:`L^2` Regularization dampings (must have the same number of elements
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
        :py:func:`scipy.sparse.linalg.lsqr` solver used to solve the first
        subproblem in the first step of the Split Bregman algorithm.

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    itn_out : :obj:`int`
        Iteration number of outer loop upon termination

    Notes
    -----
    Solve the following system of unconstrained, regularized equations
    given the operator :math:`\mathbf{Op}` and a set of mixed norm
    (:math:`L^2` and :math:`L^1`)
    regularization terms :math:`\mathbf{R}_{2,i}` and
    :math:`\mathbf{R}_{1,i}`, respectively:

    .. math::
        J = \frac{\mu}{2} \|\textbf{d} - \textbf{Op}\,\textbf{x} \|_2^2 +
        \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{d}_{\mathbf{R}_{2,i}} -
        \mathbf{R}_{2,i} \textbf{x} \|_2^2 +
        \sum_i \epsilon_{\mathbf{R}_{1,i}} \| \mathbf{R}_{1,i} \textbf{x} \|_1

    where :math:`\mu` is the reconstruction damping, :math:`\epsilon_{\mathbf{R}_{2,i}}`
    are the damping factors used to weight the different :math:`L^2` regularization
    terms of the cost function and :math:`\epsilon_{\mathbf{R}_{1,i}}`
    are the damping factors used to weight the different :math:`L^1` regularization
    terms of the cost function.

    The generalized Split-Bergman algorithm [1]_ is used to solve such cost
    function: the algorithm is composed of a sequence of unconstrained
    inverse problems and Bregman updates.

    The original system of equations is initially converted into a constrained
    problem:

    .. math::
        J = \frac{\mu}{2} \|\textbf{d} - \textbf{Op}\,\textbf{x}\|_2^2 +
        \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{d}_{\mathbf{R}_{2,i}} -
        \mathbf{R}_{2,i} \textbf{x}\|_2^2 +
        \sum_i \| \textbf{y}_i \|_1 \quad \text{subject to} \quad
        \textbf{y}_i = \mathbf{R}_{1,i} \textbf{x} \quad \forall i

    and solved as follows:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \begin{align}
        (\textbf{x}^{k+1}, \textbf{y}_i^{k+1}) =
        \argmin_{\mathbf{x}, \mathbf{y}_i}
        \|\textbf{d} - \textbf{Op}\,\textbf{x}\|_2^2
        &+ \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{d}_{\mathbf{R}_{2,i}} -
        \mathbf{R}_{2,i} \textbf{x}\|_2^2 \\
        &+ \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{1,i}} \|\textbf{y}_i -
        \mathbf{R}_{1,i} \textbf{x} - \textbf{b}_i^k\|_2^2 \\
        &+ \sum_i \| \textbf{y}_i \|_1
        \end{align}

    .. math::
        \textbf{b}_i^{k+1}=\textbf{b}_i^k +
        (\mathbf{R}_{1,i} \textbf{x}^{k+1} - \textbf{y}^{k+1})

    The :py:func:`scipy.sparse.linalg.lsqr` solver and a fast shrinkage
    algorithm are used within a inner loop to solve the first step. The entire
    procedure is repeated ``niter_outer`` times until convergence.

    .. [1] Goldstein T. and Osher S., "The Split Bregman Method for
       L1-Regularized Problems", SIAM J. on Scientific Computing, vol. 2(2),
       pp. 323-343. 2008.

    """
    ncp = get_array_module(data)

    if show:
        tstart = time.time()
        print(
            "Split-Bregman optimization\n"
            "---------------------------------------------------------\n"
            f"The Operator Op has {Op.shape[0]} rows and {Op.shape[1]} cols\n"
            f"niter_outer = {niter_outer:3d}     niter_inner = {niter_inner:3d}   tol = {tol:2.2e}\n"
            f"mu = {mu:2.2e}         epsL1 = {epsRL1s}\t  epsL2 = {epsRL2s}     "
        )
        print("---------------------------------------------------------\n")
        head1 = "   Itn          x[0]           r2norm          r12norm"
        print(head1)

    # L1 regularizations
    nregsL1 = len(RegsL1)
    b = [ncp.zeros(RegL1.shape[0], dtype=Op.dtype) for RegL1 in RegsL1]
    d = b.copy()

    # L2 regularizations
    nregsL2 = 0 if RegsL2 is None else len(RegsL2)
    if nregsL2 > 0:
        Regs = RegsL2 + RegsL1
        if dataregsL2 is None:
            dataregsL2 = [ncp.zeros(Reg.shape[0], dtype=Op.dtype) for Reg in RegsL2]
    else:
        Regs = RegsL1
        dataregsL2 = []

    # Rescale dampings
    epsRs = [
        np.sqrt(epsRL2s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(nregsL2)
    ] + [np.sqrt(epsRL1s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(nregsL1)]
    xinv = ncp.zeros(Op.shape[1], dtype=Op.dtype) if x0 is None else x0
    xold = ncp.full(Op.shape[1], ncp.inf, dtype=Op.dtype)

    itn_out = 0
    while ncp.linalg.norm(xinv - xold) > tol and itn_out < niter_outer:
        xold = xinv
        for _ in range(niter_inner):
            # Regularized problem
            dataregs = dataregsL2 + [d[ireg] - b[ireg] for ireg in range(nregsL1)]
            xinv = RegularizedInversion(
                Op,
                Regs,
                data,
                dataregs=dataregs,
                epsRs=epsRs,
                x0=x0 if restart else xinv,
                **kwargs_lsqr,
            )
            # Shrinkage
            d = [
                _softthreshold(RegsL1[ireg] * xinv + b[ireg], epsRL1s[ireg])
                for ireg in range(nregsL1)
            ]
        # Bregman update
        b = [b[ireg] + tau * (RegsL1[ireg] * xinv - d[ireg]) for ireg in range(nregsL1)]
        itn_out += 1

        if show:
            costdata = mu / 2.0 * ncp.linalg.norm(data - Op.matvec(xinv)) ** 2
            costregL2 = (
                0
                if RegsL2 is None
                else [
                    epsRL2 * ncp.linalg.norm(dataregL2 - RegL2.matvec(xinv)) ** 2
                    for epsRL2, RegL2, dataregL2 in zip(epsRL2s, RegsL2, dataregsL2)
                ]
            )
            costregL1 = [
                ncp.linalg.norm(RegL1.matvec(xinv), ord=1)
                for epsRL1, RegL1 in zip(epsRL1s, RegsL1)
            ]
            cost = (
                costdata + ncp.sum(ncp.array(costregL2)) + ncp.sum(ncp.array(costregL1))
            )
            msg = (
                f"{ncp.abs(itn_out):6g}  {ncp.real(xinv[0]):12.5e}       "
                f"{costdata:10.3e}        {cost:9.3e}"
            )
            print(msg)

    if show:
        print(
            f"\nIterations = {itn_out}        Total time (s) = {time.time() - tstart:.2f}"
        )
        print("---------------------------------------------------------\n")
    return xinv, itn_out
