import numpy as np

from pylops.basicoperators import Diagonal
from pylops.optimization.leastsquares import NormalEquationsInversion


def IRLS(Op, data, nouter, threshR=False, epsR=1e-10,
         epsI=1e-10, x0=None, tolIRLS=1e-10,
         returnhistory=False, **kwargs_cg):
    r"""Iteratively reweighted least squares.

    Solve an optimization problem with :math:`L1` cost function given the operator
    ``Op`` and data ``y``. The cost function is minimized by iteratively
    solving a weighted least squares problem with the weight at iteration
    :math:`i` being based on the data residual at iteration :math:`i+1`.

    The IRLS solver is robust to *outliers* since the L1 norm given less
    importance to large residuals than L2 norm does.

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
        Number of outer iterations
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
