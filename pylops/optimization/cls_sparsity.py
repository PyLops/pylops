__all__ = ["IRLS"]
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
from pylops.utils.backend import get_array_module, get_module_name
from pylops.utils.decorators import disable_ndarray_multiplication

try:
    from spgl1 import spgl1 as ext_spgl1
except ModuleNotFoundError:
    ext_spgl1 = None
    spgl1_message = "Spgl1 not installed. " 'Run "pip install spgl1".'
except Exception as e:
    ext_spgl1 = None
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
        Thresholded vector

    """
    thresh = np.percentile(np.abs(x), perc)
    # return _halfthreshold(x, (2. / 3. * thresh) ** (1.5))
    return _halfthreshold(x, (4.0 / 54 ** (1.0 / 3.0) * thresh) ** 1.5)


class IRLS(Solver):
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

    Raises
    ------
    NotImplementedError
        If ``kind`` is different from model or data

    Notes
    -----
    *Data IRLS* solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_1

    by a set of outer iterations which require to repeatedly solve a
    weighted least squares problem of the form:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \mathbf{x}^{(i+1)} = \argmin_\mathbf{x} \|\mathbf{y} -
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
    operator :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
        J = \|\mathbf{x}\|_1 \quad \text{subject to} \quad
        \mathbf{y} = \mathbf{Op}\,\mathbf{x}

    by a set of outer iterations which require to repeatedly solve a
    weighted least squares problem of the form [2]_:

    .. math::
        \mathbf{x}^{(i+1)} = \operatorname*{arg\,min}_\mathbf{x}
        \|\mathbf{x}\|_{2, \mathbf{R}^{(i)}}^2 \quad \text{subject to} \quad
        \mathbf{y} = \mathbf{Op}\,\mathbf{x}

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

    def _print_setup(self, xcomplex=False):
        self._print_solver(f" ({self.kind})")

        strpar = f"threshR = {self.threshR}\tepsR = {self.epsR}\tepsI = {self.epsI}"
        if self.nouter is not None:
            strpar1 = f"tolIRL = {self.nouter}\tnouter = {self.nouter}"
        else:
            strpar1 = f"tolIRL = {self.nouter}"
        print(strpar)
        print(strpar1)
        print("-" * 80)
        if not xcomplex:
            head1 = "    Itn           x[0]              r2norm"
        else:
            head1 = "    Itn              x[0]                  r2norm"
        print(head1)

    def _print_step(self, x):
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}"
        str1 = f"{self.iiter:6g}        " + strx
        str2 = f"         {self.rnorm:10.3e}"
        print(str1 + str2)

    def setup(
        self,
        y,
        nouter=None,
        threshR=False,
        epsR=1e-10,
        epsI=1e-10,
        tolIRLS=1e-10,
        kind="data",
        show=False,
    ):
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
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
            Display setup log

        """
        self.y = y
        self.nouter = nouter
        self.threshR = threshR
        self.epsR = epsR
        self.epsI = epsI
        self.tolIRLS = tolIRLS
        self.kind = kind
        self.ncp = get_array_module(y)
        self.iiter = 0

        # choose step to use
        if self.kind == "data":
            self._step = self._step_data
        elif self.kind == "model":
            self._step = self._step_model
            self.Iop = Identity(y.size, dtype=y.dtype)
        else:
            raise NotImplementedError("kind must be model or data")

        # print setup
        if show:
            self._print_setup()

    def _step_data(self, x, **kwargs_solver):
        r"""Run one step of solver with L1 data term"""
        if self.iiter == 0:
            # first iteration (unweighted least-squares)
            x = normal_equations_inversion(
                self.Op, self.y, None, epsI=self.epsI, **kwargs_solver
            )[0]
        else:
            # other iterations (weighted least-squares)
            if self.threshR:
                self.rw = 1.0 / self.ncp.maximum(self.ncp.abs(self.r), self.epsR)
            else:
                self.rw = 1.0 / (self.ncp.abs(self.r) + self.epsR)
            self.rw = self.rw / self.rw.max()
            R = Diagonal(self.rw)
            x = normal_equations_inversion(
                self.Op, self.y, [], Weight=R, epsI=self.epsI, **kwargs_solver
            )[0]
        return x

    def _step_model(self, x, **kwargs_solver):
        r"""Run one step of solver with L1 model term"""
        if self.iiter == 0:
            # first iteration (unweighted least-squares)
            if self.ncp == np:
                x = (
                    self.Op.H
                    @ lsqr(
                        self.Op @ self.Op.H + (self.epsI**2) * self.Iop,
                        self.y,
                        **kwargs_solver,
                    )[0]
                )
            else:
                x = (
                    self.Op.H
                    @ cgls(
                        self.Op @ self.Op.H + (self.epsI**2) * self.Iop,
                        self.y,
                        self.ncp.zeros(int(self.Op.shape[0]), dtype=self.Op.dtype),
                        **kwargs_solver,
                    )[0]
                )
        else:
            # other iterations (weighted least-squares)
            self.rw = np.abs(x)
            self.rw = self.rw / self.rw.max()
            R = Diagonal(self.rw, dtype=self.rw.dtype)
            if self.ncp == np:
                x = (
                    R
                    @ self.Op.H
                    @ lsqr(
                        self.Op @ R @ self.Op.H + self.epsI**2 * self.Iop,
                        self.y,
                        **kwargs_solver,
                    )[0]
                )
            else:
                x = (
                    R
                    @ self.Op.H
                    @ cgls(
                        self.Op @ R @ self.Op.H + self.epsI**2 * self.Iop,
                        self.y,
                        self.ncp.zeros(int(self.Op.shape[0]), dtype=self.Op.dtype),
                        **kwargs_solver,
                    )[0]
                )
        return x

    def step(self, x, show=False, **kwargs_solver):
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of ISTA
        show : :obj:`bool`, optional
            Display iteration log
        **kwargs_solver
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.cg` solver for data IRLS and
            :py:func:`scipy.sparse.linalg.lsqr` solver for model IRLS when using
            numpy data(or :py:func:`pylops.optimization.solver.cg` and
            :py:func:`pylops.optimization.solver.cgls` when using cupy data)

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector

        """
        # update model
        x = self._step(x, **kwargs_solver)

        # compute residual
        self.r = self.y - self.Op * x
        self.rnorm = self.ncp.linalg.norm(self.r)

        self.iiter += 1
        if show:
            self._print_step(x)
        return x

    def run(self, x, nouter=10, show=False, itershow=[10, 10, 10], **kwargs_solver):
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of IRLS
        nouter : :obj:`int`, optional
            Number of outer iterations.
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        **kwargs_solver
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.cg` solver for data IRLS and
            :py:func:`scipy.sparse.linalg.lsqr` solver for model IRLS when using
            numpy data(or :py:func:`pylops.optimization.solver.cg` and
            :py:func:`pylops.optimization.solver.cgls` when using cupy data)

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        nouter = nouter if self.nouter is None else self.nouter
        if x is not None:
            self.x0 = x.copy()
            self.y = self.y - self.Op * x
        # choose xold to ensure tolerance test is passed initially
        xold = x.copy() + np.inf
        while self.iiter < nouter and self.ncp.linalg.norm(x - xold) >= self.tolIRLS:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or nouter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            xold = x.copy()
            x = self.step(x, showstep, **kwargs_solver)
            self.callback(x)

        # adding initial guess
        if hasattr(self, "x0"):
            x = self.x0 + x
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
        self.nouter = self.iiter
        if show:
            self._print_finalize()

    def solve(
        self,
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
        **kwargs_solver,
    ):
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[N \times 1]`. If ``None``, initialize
            internally as zero vector
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
            Display setup log
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        **kwargs_solver
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.cg` solver for data IRLS and
            :py:func:`scipy.sparse.linalg.lsqr` solver for model IRLS when using
            numpy data(or :py:func:`pylops.optimization.solver.cg` and
            :py:func:`pylops.optimization.solver.cgls` when using cupy data)

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[N \times 1]`

        """
        self.setup(
            y=y,
            threshR=threshR,
            epsR=epsR,
            epsI=epsI,
            tolIRLS=tolIRLS,
            kind=kind,
            show=show,
        )
        if x0 is None:
            x0 = self.ncp.zeros(self.Op.shape[1], dtype=self.y.dtype)
        x = self.run(x0, nouter=nouter, show=show, itershow=itershow, **kwargs_solver)
        self.finalize(show)
        return x, self.nouter


class OMP(Solver):
    r"""Orthogonal Matching Pursuit (OMP).

    Solve an optimization problem with :math:`L_0` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert

    See Also
    --------
    ISTA: Iterative Shrinkage-Thresholding Algorithm (ISTA).
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    SPGL1: Spectral Projected-Gradient for L1 norm (SPGL1).
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
            \|\mathbf{x}\|_0 \quad  \text{subject to} \quad
            \|\mathbf{Op}\,\mathbf{x}-\mathbf{y}\|_2^2 \leq \sigma^2,

    using Orthogonal Matching Pursuit (OMP). This is a very
    simple iterative algorithm which applies the following step:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \DeclareMathOperator*{\argmax}{arg\,max}
        \Lambda_k = \Lambda_{k-1} \cup \left\{\argmax_j
        \left|\mathbf{Op}_j^H\,\mathbf{r}_k\right| \right\} \\
        \mathbf{x}_k = \argmin_{\mathbf{x}}
        \left\|\mathbf{Op}_{\Lambda_k}\,\mathbf{x} - \mathbf{y}\right\|_2^2

    Note that by choosing ``niter_inner=0`` the basic Matching Pursuit (MP)
    algorithm is implemented instead. In other words, instead of solving an
    optimization at each iteration to find the best :math:`\mathbf{x}` for the
    currently selected basis functions, the vector :math:`\mathbf{x}` is just
    updated at the new basis function by taking directly the value from
    the inner product :math:`\mathbf{Op}_j^H\,\mathbf{r}_k`.

    In this case it is highly recommended to provide a normalized basis
    function. If different basis have different norms, the solver is likely
    to diverge. Similar observations apply to OMP, even though mild unbalancing
    between the basis is generally properly handled.

    """

    def _print_setup(self, xcomplex=False):
        self._print_solver("(Only MP)" if self.niter_inner == 0 else "", nbar=55)

        strpar = (
            f"sigma = {self.sigma:.2e}\tniter_outer = {self.niter_outer}\n"
            f"niter_inner = {self.niter_inner}\tnormalization={self.normalizecols}"
        )
        print(strpar)
        print("-" * 55)
        if not xcomplex:
            head1 = "    Itn           x[0]              r2norm"
        else:
            head1 = "    Itn              x[0]                  r2norm"
        print(head1)

    def _print_step(self, x):
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}"
        str1 = f"{self.iiter:6g}        " + strx
        str2 = f"         {self.cost[-1]:10.3e}"
        print(str1 + str2)

    def setup(
        self,
        y,
        niter_outer=10,
        niter_inner=40,
        sigma=1e-4,
        normalizecols=False,
        show=False,
    ):
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
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
            Display setup log

        """
        self.Op = LinearOperator(self.Op)
        self.y = y
        self.niter_outer = niter_outer
        self.niter_inner = niter_inner
        self.sigma = sigma
        self.normalizecols = normalizecols
        self.ncp = get_array_module(y)

        # find normalization factor for each column
        if self.normalizecols:
            ncols = self.Op.shape[1]
            self.norms = self.ncp.zeros(ncols)
            for icol in range(ncols):
                unit = self.ncp.zeros(ncols, dtype=self.Op.dtype)
                unit[icol] = 1
                self.norms[icol] = np.linalg.norm(self.Op.matvec(unit))

        # create variables to track the residual norm and iterations
        self.res = self.y.copy()
        self.cost = [
            np.linalg.norm(self.y),
        ]
        self.iiter = 0

        if show:
            self._print_setup()

    def step(self, x, cols, show=False):
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`list` or :obj:`np.ndarray`
            Current model vector to be updated by a step of OMP
        cols : :obj:`list`
            Current list of chosen elements of vector x to be updated by a step of OMP
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector
        cols : :obj:`list`
            Current list of chosen elements

        """
        # compute inner products
        cres = self.Op.rmatvec(self.res)
        cres_abs = np.abs(cres)
        if self.normalizecols:
            cres_abs = cres_abs / self.norms
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
        if self.niter_inner == 0:
            # MP update
            Opcol = self.Op.apply_columns(
                [
                    int(imax),
                ]
            )
            self.res -= Opcol.matvec(cres[imax] * self.ncp.ones(1))
            if addnew:
                x.append(cres[imax])
            else:
                x[imax_in_cols] += cres[imax]
        else:
            # OMP update
            Opcol = self.Op.apply_columns(cols)
            if self.ncp == np:
                x = lsqr(Opcol, self.y, iter_lim=self.niter_inner)[0]
            else:
                x = cgls(
                    Opcol,
                    self.y,
                    self.ncp.zeros(int(Opcol.shape[1]), dtype=Opcol.dtype),
                    niter=self.niter_inner,
                )[0]
            self.res = self.y - Opcol.matvec(x)

        self.iiter += 1
        self.cost.append(np.linalg.norm(self.res))
        if show:
            self._print_step(x)
        return x, cols

    def run(self, x, cols, show=False, itershow=[10, 10, 10]):
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of IRLS
        cols : :obj:`list`
            Current list of chosen elements of vector x to be updated by a step of OMP
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
        cols : :obj:`list`
            Current list of chosen elements

        """
        while self.iiter < self.niter_outer and self.cost[self.iiter] > self.sigma:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or self.niter_outer - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, cols = self.step(x, cols, showstep)
            self.callback(x)
        return x, cols

    def finalize(self, x, cols, show=False):
        r"""Finalize solver

        Parameters
        ----------
        x : :obj:`list` or :obj:`np.ndarray`
            Current model vector to be updated by a step of OMP
        cols : :obj:`list`
            Current list of chosen elements of vector x to be updated by a step of OMP
        show : :obj:`bool`, optional
            Display finalize log

        Returns
        -------
        xfin : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        self.cost = np.array(self.cost)
        self.nouter = self.iiter

        xfin = self.ncp.zeros(int(self.Op.shape[1]), dtype=self.Op.dtype)
        xfin[cols] = self.ncp.array(x)
        if show:
            self._print_finalize(nbar=55)
        return xfin

    def solve(
        self,
        y,
        niter_outer=10,
        niter_inner=40,
        sigma=1e-4,
        normalizecols=False,
        show=False,
        itershow=[10, 10, 10],
    ):
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
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
        self.setup(
            y,
            niter_outer=niter_outer,
            niter_inner=niter_inner,
            sigma=sigma,
            normalizecols=normalizecols,
            show=show,
        )
        x, cols = [], []
        x, cols = self.run(x, cols, show=show, itershow=itershow)
        x = self.finalize(x, cols, show)
        return x, self.nouter, self.cost


class ISTA(Solver):
    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solve an optimization problem with :math:`L_p, \; p=0, 0.5, 1`
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
    :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{x}\|_p

    or the analysis problem:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{SOp}^H\,\mathbf{x}\|_p

    if ``SOp`` is provided. Note that in the first case, ``SOp`` should be
    assimilated in the modelling operator (i.e., ``Op=GOp * SOp``).

    The Iterative Shrinkage-Thresholding Algorithms (ISTA) [1]_ is used, where
    :math:`p=0, 0.5, 1`. This is a very simple iterative algorithm which
    applies the following step:

    .. math::
        \mathbf{x}^{(i+1)} = T_{(\epsilon \alpha /2, p)} \left(\mathbf{x}^{(i)} +
        \alpha\,\mathbf{Op}^H \left(\mathbf{y} - \mathbf{Op}\,\mathbf{x}^{(i)}\right)\right)

    or

    .. math::
        \mathbf{x}^{(i+1)} = \mathbf{SOp}\,\left\{T_{(\epsilon \alpha /2, p)}
        \mathbf{SOp}^H\,\left(\mathbf{x}^{(i)} + \alpha\,\mathbf{Op}^H \left(\mathbf{y} -
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
        self._print_solver(f" ({self.threshkind} thresholding)")
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
        print("-" * 80)
        print(head1)

    def _print_step(self, x, costdata, costreg, xupdate):
        strx = (
            f"  {x[0]:1.2e}   " if np.iscomplexobj(x) else f"     {x[0]:11.4e}        "
        )
        msg = (
            f"{self.iiter:6g} "
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
        eigsdict=None,
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
        eigsdict : :obj:`dict`, optional
            Dictionary of parameters to be passed to :func:`pylops.LinearOperator.eigs` method
            when computing the maximum eigenvalue
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
        self.eigsdict = {} if eigsdict is None else eigsdict
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
        if alpha is not None:
            self.alpha = alpha
        elif not hasattr(self, "alpha"):
            if not isinstance(self.Op, LinearOperator):
                self.Op = LinearOperator(self.Op, explicit=False)
            # compute largest eigenvalues of Op^H * Op
            Op1 = LinearOperator(self.Op.H * self.Op, explicit=False)
            if get_module_name(self.ncp) == "numpy":
                maxeig = np.abs(
                    Op1.eigs(
                        neigs=1,
                        symmetric=True,
                        **self.eigsdict,
                    )[0]
                )
            else:
                maxeig = np.abs(
                    power_iteration(
                        Op1,
                        dtype=Op1.dtype,
                        backend="cupy",
                        **self.eigsdict,
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
        # store old vector
        xold = x.copy()
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
        x_unthesh = x + grad
        if self.SOp is not None:
            x_unthesh = self.SOp.H @ x_unthesh
        if self.perc is None:
            x = self.threshf(x_unthesh, self.decay[self.iiter] * self.thresh)
        else:
            x = self.threshf(x_unthesh, 100 - self.perc)
        if self.SOp is not None:
            x = self.SOp @ x

        # model update
        xupdate = np.linalg.norm(x - xold)

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
        eigsdict=None,
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
        eigsdict : :obj:`dict`, optional
            Dictionary of parameters to be passed to :func:`pylops.LinearOperator.eigs` method
            when computing the maximum eigenvalue
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
            eigsdict=eigsdict,
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

    Solve an optimization problem with :math:`L_p, \; p=0, 0.5, 1`
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
    :math:`\mathbf{Op}` and the data :math:`\mathbf{y}`:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
            \epsilon \|\mathbf{x}\|_p

    or the analysis problem:

    .. math::
        J = \|\mathbf{y} - \mathbf{Op}\,\mathbf{x}\|_2^2 +
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
        # store old vector
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
            \left\|\mathbf{Op}\,\mathbf{S}^H\mathbf{x}-\mathbf{y}\right\|_2^2
            \leq \sigma,

    while the second problem is the *ℓ₁-regularized least-squares or LASSO*
    problem and its cost function is

        .. math::
            \left\|\mathbf{Op}\,\mathbf{S}^H\mathbf{x}-\mathbf{y}\right\|_2^2
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
        print("-" * 80)

    def _print_finalize(self):
        print(f"\nTotal time (s) = {self.telapsed:.2f}")
        print("-" * 80 + "\n")

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
        if ext_spgl1 is None:
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
        pinv, _, _, info = ext_spgl1(
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


class SplitBregman(Solver):
    r"""Split Bregman for mixed L2-L1 norms.

    Solve an unconstrained system of equations with mixed :math:`L_2` and :math:`L_1`
    regularization terms given the operator ``Op``, a list of :math:`L_1`
    regularization terms ``RegsL1``, and an optional list of :math:`L_2`
    regularization terms ``RegsL2``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert

    Notes
    -----
    Solve the following system of unconstrained, regularized equations
    given the operator :math:`\mathbf{Op}` and a set of mixed norm
    (:math:`L^2` and :math:`L_1`)
    regularization terms :math:`\mathbf{R}_{2,i}` and
    :math:`\mathbf{R}_{1,i}`, respectively:

    .. math::
        J = \frac{\mu}{2} \|\textbf{y} - \textbf{Op}\,\textbf{x} \|_2^2 +
        \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{y}_{\mathbf{R}_{2,i}} -
        \mathbf{R}_{2,i} \textbf{x} \|_2^2 +
        \sum_i \epsilon_{\mathbf{R}_{1,i}} \| \mathbf{R}_{1,i} \textbf{x} \|_1

    where :math:`\mu` is the reconstruction damping, :math:`\epsilon_{\mathbf{R}_{2,i}}`
    are the damping factors used to weight the different :math:`L^2` regularization
    terms of the cost function and :math:`\epsilon_{\mathbf{R}_{1,i}}`
    are the damping factors used to weight the different :math:`L_1` regularization
    terms of the cost function.

    The generalized Split-Bergman algorithm [1]_ is used to solve such cost
    function: the algorithm is composed of a sequence of unconstrained
    inverse problems and Bregman updates.

    The original system of equations is initially converted into a constrained
    problem:

    .. math::
        J = \frac{\mu}{2} \|\textbf{y} - \textbf{Op}\,\textbf{x}\|_2^2 +
        \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{y}_{\mathbf{R}_{2,i}} -
        \mathbf{R}_{2,i} \textbf{x}\|_2^2 +
        \sum_i \| \textbf{y}_i \|_1 \quad \text{subject to} \quad
        \textbf{y}_i = \mathbf{R}_{1,i} \textbf{x} \quad \forall i

    and solved as follows:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \begin{align}
        (\textbf{x}^{k+1}, \textbf{y}_i^{k+1}) =
        \argmin_{\mathbf{x}, \mathbf{y}_i}
        \|\textbf{y} - \textbf{Op}\,\textbf{x}\|_2^2
        &+ \frac{1}{2}\sum_i \epsilon_{\mathbf{R}_{2,i}} \|\mathbf{y}_{\mathbf{R}_{2,i}} -
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

    def _print_setup(self, xcomplex=False):
        self._print_solver(nbar=65)

        strpar = (
            f"niter_outer = {self.niter_outer:3d}     niter_inner = {self.niter_inner:3d}   tol = {self.tol:2.2e}\n"
            f"mu = {self.mu:2.2e}         epsL1 = {self.epsRL1s}\t  epsL2 = {self.epsRL2s}"
        )
        print(strpar)
        print("-" * 65)
        if not xcomplex:
            head1 = "    Itn       x[0]           r2norm           r12norm"
        else:
            head1 = "    Itn          x[0]            r2norm           r12norm"
        print(head1)

    def _print_step(self, x):
        strx = f"{x[0]:1.2e}   " if np.iscomplexobj(x) else f"{x[0]:11.4e}      "
        str1 = f"{self.iiter:6g}    " + strx
        str2 = f"{self.costdata:10.3e}        {self.costtot:9.3e}"
        print(str1 + str2)

    def setup(
        self,
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
    ):
        r"""Setup solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        RegsL1 : :obj:`list`
            :math:`L_1` regularization operators
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
        niter_outer : :obj:`int`, optional
            Number of iterations of outer loop
        niter_inner : :obj:`int`, optional
            Number of iterations of inner loop of first step of the Split Bregman
            algorithm. A small number of iterations is generally sufficient and
            for many applications optimal efficiency is obtained when only one
            iteration is performed.
        RegsL2 : :obj:`list`, optional
            Additional :math:`L^2` regularization operators
            (if ``None``, :math:`L^2` regularization is not added to the problem)
        dataregsL2 : :obj:`list`, optional
            :math:`L^2` Regularization data (must have the same number of elements
            of ``RegsL2`` or equal to ``None`` to use a zero data for every
            regularization operator in ``RegsL2``)
        mu : :obj:`float`, optional
             Data term damping
        epsRL1s : :obj:`list`
             :math:`L_1` Regularization dampings (must have the same number of elements
             as ``RegsL1``)
        epsRL2s : :obj:`list`
             :math:`L^2` Regularization dampings (must have the same number of elements
             as ``RegsL2``)
        tol : :obj:`float`, optional
            Tolerance. Stop outer iterations if difference between inverted model
            at subsequent iterations is smaller than ``tol``
        tau : :obj:`float`, optional
            Scaling factor in the Bregman update (must be close to 1)
        restart : :obj:`bool`, optional
            Initialize the unconstrained inverse problem in inner loop with
            the initial guess (``True``) or with the last estimate (``False``).
            Note that when this is set to ``True``, the ``x0`` provided in the setup will
            be used in all iterations.
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`np.ndarray`
            Initial guess of size :math:`[N \times 1]`

        """
        self.y = y
        self.RegsL1 = RegsL1
        self.niter_outer = niter_outer
        self.niter_inner = niter_inner
        self.RegsL2 = RegsL2
        self.dataregsL2 = dataregsL2
        self.dataregsL2 = dataregsL2
        self.mu = mu
        self.epsRL1s = epsRL1s
        self.epsRL2s = epsRL2s
        self.tol = tol
        self.tau = tau
        self.restart = restart
        self.ncp = get_array_module(y)

        # L1 regularizations
        self.nregsL1 = len(RegsL1)
        self.b = [
            self.ncp.zeros(RegL1.shape[0], dtype=self.Op.dtype) for RegL1 in RegsL1
        ]
        self.d = self.b.copy()

        # L2 regularizations
        self.nregsL2 = 0 if RegsL2 is None else len(RegsL2)
        if self.nregsL2 > 0:
            self.Regs = RegsL2 + RegsL1
            if dataregsL2 is None:
                self.dataregsL2 = [
                    self.ncp.zeros(Reg.shape[0], dtype=self.Op.dtype) for Reg in RegsL2
                ]
        else:
            self.Regs = RegsL1
            self.dataregsL2 = []

        # Rescale dampings
        self.epsRs = [
            np.sqrt(epsRL2s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(self.nregsL2)
        ] + [
            np.sqrt(epsRL1s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(self.nregsL1)
        ]

        self.x0 = x0
        x = self.ncp.zeros(self.Op.shape[1], dtype=self.Op.dtype) if x0 is None else x0

        # create variables to track the residual norm and iterations
        self.cost = []
        self.iiter = 0

        if show:
            self._print_setup(np.iscomplexobj(x))
        return x

    def step(self, x, show=False, show_inner=False, **kwargs_lsqr):
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`list` or :obj:`np.ndarray`
            Current model vector to be updated by a step of OMP
        show_inner : :obj:`bool`, optional
            Display inner iteration logs of lsqr
        show : :obj:`bool`, optional
            Display iteration log
        **kwargs_lsqr
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.lsqr` solver used to solve the first
            subproblem in the first step of the Split Bregman algorithm.

        Returns
        -------
        x : :obj:`np.ndarray`
            Updated model vector

        """
        for _ in range(self.niter_inner):
            # regularized problem
            dataregs = self.dataregsL2 + [
                self.d[ireg] - self.b[ireg] for ireg in range(self.nregsL1)
            ]
            x = regularized_inversion(
                self.Op,
                self.y,
                self.Regs,
                dataregs=dataregs,
                epsRs=self.epsRs,
                x0=self.x0 if self.restart else x,
                show=show_inner,
                **kwargs_lsqr,
            )[0]
            # Shrinkage
            self.d = [
                _softthreshold(self.RegsL1[ireg] * x + self.b[ireg], self.epsRL1s[ireg])
                for ireg in range(self.nregsL1)
            ]

        # Bregman update
        self.b = [
            self.b[ireg] + self.tau * (self.RegsL1[ireg] * x - self.d[ireg])
            for ireg in range(self.nregsL1)
        ]

        # compute residual norms
        self.costdata = (
            self.mu / 2.0 * self.ncp.linalg.norm(self.y - self.Op.matvec(x)) ** 2
        )
        self.costregL2 = (
            0
            if self.RegsL2 is None
            else [
                epsRL2 * self.ncp.linalg.norm(dataregL2 - RegL2.matvec(x)) ** 2
                for epsRL2, RegL2, dataregL2 in zip(
                    self.epsRL2s, self.RegsL2, self.dataregsL2
                )
            ]
        )
        self.costregL1 = [
            self.ncp.linalg.norm(RegL1.matvec(x), ord=1)
            for epsRL1, RegL1 in zip(self.epsRL1s, self.RegsL1)
        ]
        self.costtot = (
            self.costdata
            + self.ncp.sum(self.ncp.array(self.costregL2))
            + self.ncp.sum(self.ncp.array(self.costregL1))
        )

        # update history parameters
        self.iiter += 1
        self.cost.append(self.costtot)
        if show:
            self._print_step(x)
        return x

    def run(
        self, x, show=False, itershow=[10, 10, 10], show_inner=False, **kwargs_lsqr
    ):
        r"""Run solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of IRLS
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        show_inner : :obj:`bool`, optional
            Display inner iteration logs of lsqr
        **kwargs_lsqr
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.lsqr` solver used to solve the first
            subproblem in the first step of the Split Bregman algorithm.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        xold = x.copy() + 1.1 * self.tol
        while (
            self.ncp.linalg.norm(x - xold) > self.tol and self.iiter < self.niter_outer
        ):
            xold = x.copy()
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or self.niter_outer - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x = self.step(x, showstep, show_inner, **kwargs_lsqr)
            self.callback(x)
        return x

    def finalize(self, show=False):
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        Returns
        -------
        xfin : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        self.cost = np.array(self.cost)

        if show:
            self._print_finalize(nbar=65)

    def solve(
        self,
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
        **kwargs_lsqr,
    ):
        r"""Run entire solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        RegsL1 : :obj:`list`
            :math:`L_1` regularization operators
        x0 : :obj:`np.ndarray`, optional
            Initial guess of size :math:`[M \times 1]`. If ``None``, initialize
            internally as zero vector
        niter_outer : :obj:`int`, optional
            Number of iterations of outer loop
        niter_inner : :obj:`int`, optional
            Number of iterations of inner loop of first step of the Split Bregman
            algorithm. A small number of iterations is generally sufficient and
            for many applications optimal efficiency is obtained when only one
            iteration is performed.
        RegsL2 : :obj:`list`, optional
            Additional :math:`L^2` regularization operators
            (if ``None``, :math:`L^2` regularization is not added to the problem)
        dataregsL2 : :obj:`list`, optional
            :math:`L^2` Regularization data (must have the same number of elements
            of ``RegsL2`` or equal to ``None`` to use a zero data for every
            regularization operator in ``RegsL2``)
        mu : :obj:`float`, optional
             Data term damping
        epsRL1s : :obj:`list`
             :math:`L_1` Regularization dampings (must have the same number of elements
             as ``RegsL1``)
        epsRL2s : :obj:`list`
             :math:`L^2` Regularization dampings (must have the same number of elements
             as ``RegsL2``)
        tol : :obj:`float`, optional
            Tolerance. Stop outer iterations if difference between inverted model
            at subsequent iterations is smaller than ``tol``
        tau : :obj:`float`, optional
            Scaling factor in the Bregman update (must be close to 1)
        restart : :obj:`bool`, optional
            Initialize the unconstrained inverse problem in inner loop with
            the initial guess (``True``) or with the last estimate (``False``).
            Note that when this is set to ``True``, the ``x0`` provided in the setup will
            be used in all iterations.
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`list`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        show_inner : :obj:`bool`, optional
            Display inner iteration logs of lsqr
        **kwargs_lsqr
            Arbitrary keyword arguments for
            :py:func:`scipy.sparse.linalg.lsqr` solver used to solve the first
            subproblem in the first step of the Split Bregman algorithm.

        Returns
        -------
        x : :obj:`np.ndarray`
            Estimated model of size :math:`[M \times 1]`
        iiter : :obj:`int`
            Iteration number of outer loop upon termination
        cost : :obj:`numpy.ndarray`
            History of cost function through iterations

        """
        x = self.setup(
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
        )
        x = self.run(
            x, show=show, itershow=itershow, show_inner=show_inner, **kwargs_lsqr
        )
        self.finalize(show)
        return x, self.iiter, self.cost
