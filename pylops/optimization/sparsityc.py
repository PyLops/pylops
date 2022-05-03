import logging
import time

import numpy as np
from scipy.sparse.linalg import lsqr

from pylops import LinearOperator
from pylops.basicoperators import Diagonal, Identity
from pylops.optimization.basesolver import Solver
from pylops.optimization.basic import cgls
from pylops.optimization.eigs import power_iteration
from pylops.optimization.leastsquares import (
    normal_equations_inversion,
    regularized_inversion,
)
from pylops.utils.backend import get_array_module, get_module_name, to_numpy
from pylops.utils.decorators import disable_ndarray_multiplication

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
    return _hardthreshold(x, 0.5 * thresh**2)


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


class ISTA(Solver):
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``. The operator
    can be real or complex, and should ideally be either square :math:`N=M`
    or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert

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

    def _print_setup(self):
        self._print_solver(f"({self.threshkind} thresholding)")
        if self.niter is not None:
            strpar = f"eps = {self.eps:10e}\ttol = {self.tol:10e}\tniter = {self.niter}"
        else:
            strpar = f"eps = {self.eps:10e}\ttol = {self.tol:10e}"
        if self.perc is None:
            strpar1 = f"alpha = {self.alpha:10e}\tthresh = {self.thresh:10e}"
        else:
            strpar1 = f"alpha = {self.alpha:10e}\tperc = {self.perc:.1f}"
        head1 = "   Itn          x[0]              r2norm     r12norm     xupdate"
        print(strpar)
        print(strpar1)
        print("-----------------------------------------------------------")
        print(head1)

    def _print_step(self, x, costdata, costreg, xupdate):
        strx = (
            f"  {x[0]:1.2e}   " if np.iscomplexobj(x) else f"     {x[0]:11.4e}        "
        )
        msg = (
            f"{self.iiter + 1:6g} "
            + strx
            + f"{costdata:10.3e}   {costdata + costreg:9.3e}  {xupdate:10.3e}"
        )
        print(msg)
        pass

    def setup(
        self,
        y,
        x0=None,
        niter=None,
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
    ):
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
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
            Display setup log

        """
        self.y = y
        self.SOp = SOp
        self.niter = niter
        self.eps = eps
        self.alpha = alpha
        self.eigsiter = eigsiter
        self.eigstol = eigstol
        self.tol = tol
        self.threshkind = threshkind
        self.perc = perc
        self.decay = decay
        self.monitorres = monitorres

        self.ncp = get_array_module(y)

        if threshkind not in [
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
            self.threshf = _softthreshold
        elif threshkind == "hard":
            self.threshf = _hardthreshold
        elif threshkind == "half":
            self.threshf = _halfthreshold
        elif threshkind == "hard-percentile":
            self.threshf = _hardthreshold_percentile
        elif threshkind == "soft-percentile":
            self.threshf = _softthreshold_percentile
        else:
            self.threshf = _halfthreshold_percentile

        # prepare decay (if not passed)
        if perc is None and decay is None:
            self.decay = self.ncp.ones(niter)

        # step size
        if alpha is None:
            if not isinstance(self.Op, LinearOperator):
                Op = LinearOperator(self.Op, explicit=False)
            # compute largest eigenvalues of Op^H * Op
            Op1 = LinearOperator(self.Op.H * self.Op, explicit=False)
            if get_module_name(self.ncp) == "numpy":
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
                        Op1,
                        niter=eigsiter,
                        tol=eigstol,
                        dtype=Op1.dtype,
                        backend="cupy",
                    )[0]
                )
            self.alpha = 1.0 / maxeig

        # define threshold
        self.thresh = eps * self.alpha * 0.5

        # initialize model and cost function
        if x0 is None:
            if y.ndim == 1:
                x = self.ncp.zeros(int(self.Op.shape[1]), dtype=self.Op.dtype)
            else:
                x = self.ncp.zeros(
                    (int(self.Op.shape[1]), y.shape[1]), dtype=self.Op.dtype
                )
        else:
            if y.ndim != x0.ndim:
                # error for wrong dimensions
                raise ValueError("Number of columns of x0 and data are not the same")
            elif x0.shape[0] != self.Op.shape[1]:
                # error for wrong dimensions
                raise ValueError("Operator and input vector have different dimensions")
            else:
                x = x0.copy()

        # create variable to track residual
        if monitorres:
            self.normresold = np.inf

        # for fista
        self.t = 1.0

        # create variables to track the residual norm and iterations
        self.cost = []
        self.iiter = 0

        # print setup
        if show:
            self._print_setup()
        return x

    def step(self, x, show=False):
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of ISTA
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector
        xupdate : :obj:`float`
            Norm of the update

        """
        xinvold = x.copy()

        # compute residual
        res = self.y - self.Op @ x
        if self.monitorres:
            self.normres = np.linalg.norm(res)
            if self.normres > self.normresold:
                raise ValueError(
                    f"ISTA stopped at iteration {self.iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                self.normresold = self.normres

        # compute gradient
        grad = self.alpha * self.Op.H @ res

        # update inverted model
        xinv_unthesh = x + grad
        if self.SOp is not None:
            xinv_unthesh = self.SOp.H @ xinv_unthesh
        if self.perc is None:
            x = self.threshf(xinv_unthesh, self.decay[self.iiter] * self.thresh)
        else:
            x = self.threshf(xinv_unthesh, 100 - self.perc)
        if self.SOp is not None:
            x = self.SOp @ x

        # model update
        xupdate = np.linalg.norm(x - xinvold)

        costdata = 0.5 * np.linalg.norm(res) ** 2
        costreg = self.eps * np.linalg.norm(x, ord=1)
        self.cost.append(costdata + costreg)
        self.iiter += 1
        if show:
            self._print_step(x, costdata, costreg, xupdate)
        return x, xupdate

    def run(self, x, niter=None, show=False, itershow=[10, 10, 10]):
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
        xupdate = np.inf
        niter = self.niter if niter is None else niter
        while self.iiter < niter and xupdate > self.tol:
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
            x, xupdate = self.step(x, showstep)
            self.callback(x)
        if xupdate <= self.tol:
            logging.warning(
                "update smaller that tolerance for " "iteration %d", self.iiter
            )
        return x

    def finalize(self, show=False):
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
        y,
        x0=None,
        niter=None,
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
    ):
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
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

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`
        niter : :obj:`int`
            Number of effective iterations
        cost : :obj:`numpy.ndarray`, optional
            History of cost function

        """
        x = self.setup(
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
        )
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost


class FISTA(ISTA):
    r"""Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L^p, \; p=0, 0.5, 1`
    regularization, given the operator ``Op`` and data ``y``.
    The operator can be real or complex, and should ideally be either square
    :math:`N=M` or underdetermined :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert

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

    def step(self, x, z, show=False):
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of ISTA
        x : :obj:`np.ndarray`
            Current auxiliary model vector to be updated by a step of ISTA
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector
        z : :obj:`np.ndarray`
            Updated auxiliary model vector
        xupdate : :obj:`float`
            Norm of the update

        """
        xold = x.copy()

        # compute residual
        resz = self.y - self.Op @ z
        if self.monitorres:
            self.normres = np.linalg.norm(resz)
            if self.normres > self.normresold:
                raise ValueError(
                    f"ISTA stopped at iteration {self.iiter} due to "
                    "residual increasing, consider modifying "
                    "eps and/or alpha..."
                )
            else:
                self.normresold = self.normres

        # compute gradient
        grad = self.alpha * self.Op.H @ resz

        # update inverted model
        x_unthesh = z + grad
        if self.SOp is not None:
            x_unthesh = self.SOp.H @ x_unthesh
        if self.perc is None:
            x = self.threshf(x_unthesh, self.decay[self.iiter] * self.thresh)
        else:
            x = self.threshf(x_unthesh, 100 - self.perc)
        if self.SOp is not None:
            x = self.SOp @ x

        # update auxiliary coefficients
        told = self.t
        self.t = (1.0 + np.sqrt(1.0 + 4.0 * self.t**2)) / 2.0
        z = x + ((told - 1.0) / self.t) * (x - xold)

        # model update
        xupdate = np.linalg.norm(x - xold)

        costdata = 0.5 * np.linalg.norm(self.y - self.Op @ x) ** 2
        costreg = self.eps * np.linalg.norm(x, ord=1)
        self.cost.append(costdata + costreg)
        self.iiter += 1
        if show:
            self._print_step(x, costdata, costreg, xupdate)
        return x, z, xupdate

    def run(self, x, niter=None, show=False, itershow=[10, 10, 10]):
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
        z = x.copy()
        xupdate = np.inf
        niter = self.niter if niter is None else niter
        while self.iiter < niter and xupdate > self.tol:
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
            x, z, xupdate = self.step(x, z, showstep)
            self.callback(x)
        if xupdate <= self.tol:
            logging.warning(
                "update smaller that tolerance for " "iteration %d", self.iiter
            )
        return x


class SPGL1(Solver):
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
        Operator to invert of size :math:`[N \times M]`.

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

    def _print_setup(self, xcomplex=False):
        self._print_solver()
        strprec = f"SOp={self.SOp}"
        strreg = f"tau={self.tau}     sigma={self.sigma}"
        print(strprec)
        print(strreg)
        print("-----------------------------------------------------------")

    def _print_finalize(self):
        print(f"\nTotal time (s) = {self.telapsed:.2f}")
        print("-----------------------------------------------------------------\n")

    @disable_ndarray_multiplication
    def setup(self, y, SOp=None, tau=0, sigma=0, show=False):
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        SOp : :obj:`pylops.LinearOperator`, optional
            Sparsifying transform
        tau : :obj:`float`, optional
            Non-negative LASSO scalar. If different from ``0``,
            SPGL1 will solve LASSO problem
        sigma : :obj:`list`, optional
            BPDN scalar. If different from ``0``,
            SPGL1 will solve BPDN problem

        show : :obj:`bool`, optional
            Display setup log

        """
        if spgl1 is None:
            raise ModuleNotFoundError(spgl1_message)

        self.y = y
        self.SOp = SOp
        self.tau = tau
        self.sigma = sigma
        self.ncp = get_array_module(y)

        # print setup
        if show:
            self._print_setup()

    def step(self):
        raise NotImplementedError(
            "SPGL1 uses as default the"
            "spgl1.spgl1 solver, therefore the "
            "step method is not implemented. Use directly run or solve."
        )

    @disable_ndarray_multiplication
    def run(self, x, show=False, **kwargs_spgl1):
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of the solver.
            If ``None``, x is assumed to be a zero vector
        show : :obj:`bool`, optional
            Display iterations log
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

        """
        pinv, _, _, info = spgl1(
            self.Op if self.SOp is None else self.Op * self.SOp.H,
            self.y,
            tau=self.tau,
            sigma=self.sigma,
            x0=x,
            **kwargs_spgl1,
        )

        xinv = pinv.copy() if self.SOp is None else self.SOp.H * pinv
        return xinv, pinv, info

    def solve(self, y, x0=None, SOp=None, tau=0, sigma=0, show=False, **kwargs_spgl1):
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
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
        show : :obj:`bool`, optional
            Display log
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

        """
        self.setup(y=y, SOp=SOp, tau=tau, sigma=sigma, show=show)
        xinv, pinv, info = self.run(x0, show=show, **kwargs_spgl1)
        self.finalize(show)
        return xinv, pinv, info
