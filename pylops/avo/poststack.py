__all__ = [
    "PoststackLinearModelling",
    "PoststackInversion",
]

import logging
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import lsqr

from pylops import (
    FirstDerivative,
    Laplacian,
    LinearOperator,
    MatrixMult,
    SecondDerivative,
)
from pylops.optimization.basic import cgls
from pylops.optimization.leastsquares import regularized_inversion
from pylops.optimization.sparsity import splitbregman
from pylops.signalprocessing import Convolve1D
from pylops.utils import dottest as Dottest
from pylops.utils.backend import (
    get_array_module,
    get_csc_matrix,
    get_lstsq,
    get_module_name,
)
from pylops.utils.signalprocessing import convmtx, nonstationary_convmtx
from pylops.utils.typing import NDArray, ShapeLike

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _PoststackLinearModelling(
    wav,
    nt0,
    spatdims=None,
    explicit=False,
    sparse=False,
    kind="centered",
    _MatrixMult=MatrixMult,
    _Convolve1D=Convolve1D,
    _FirstDerivative=FirstDerivative,
    args_MatrixMult={},
    args_Convolve1D={},
    args_FirstDerivative={},
):
    """Post-stack linearized seismic modelling operator.

    Used to be able to provide operators from different libraries to
    PoststackLinearModelling. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    ncp = get_array_module(wav)

    # check kind is correctly selected
    if kind not in ["forward", "centered"]:
        raise NotImplementedError("%s not an available derivative kind..." % kind)

    # define dtype to be used
    dtype = wav.dtype  # ensure wav.dtype rules that of operator

    if len(wav.shape) == 2 and wav.shape[0] != nt0:
        raise ValueError("Provide 1d wavelet or 2d wavelet composed of nt0 " "wavelets")

    spatdims: Union[int, ShapeLike]
    # organize dimensions
    if spatdims is None:
        dims = (nt0,)
        spatdims = None
    elif isinstance(spatdims, int):
        dims = (nt0, spatdims)
        spatdims = (spatdims,)
    else:
        dims = (nt0,) + spatdims

    if explicit:
        # Create derivative operator
        if kind == "centered":
            D = ncp.diag(0.5 * ncp.ones(nt0 - 1, dtype=dtype), k=1) - ncp.diag(
                0.5 * ncp.ones(nt0 - 1, dtype=dtype), -1
            )
            D[0] = D[-1] = 0
        else:
            D = ncp.diag(ncp.ones(nt0 - 1, dtype=dtype), k=1) - ncp.diag(
                ncp.ones(nt0, dtype=dtype), k=0
            )
            D[-1] = 0

        # Create wavelet operator
        if len(wav.shape) == 1:
            C = convmtx(wav, nt0)[:, len(wav) // 2 : -len(wav) // 2 + 1]
        else:
            C = nonstationary_convmtx(wav, nt0, hc=wav.shape[1] // 2, pad=(nt0, nt0))
        # Combine operators
        M = ncp.dot(C, D)
        if sparse:
            M = get_csc_matrix(wav)(M)
        Pop = _MatrixMult(M, otherdims=spatdims, dtype=dtype, **args_MatrixMult)
    else:
        # Create wavelet operator
        if len(wav.shape) == 1:
            Cop = _Convolve1D(
                dims,
                h=wav,
                offset=len(wav) // 2,
                axis=0,
                dtype=dtype,
                **args_Convolve1D
            )
        else:
            Cop = _MatrixMult(
                nonstationary_convmtx(wav, nt0, hc=wav.shape[1] // 2, pad=(nt0, nt0)),
                otherdims=spatdims,
                dtype=dtype,
                **args_MatrixMult
            )
        # Create derivative operator
        Dop = _FirstDerivative(
            dims, axis=0, sampling=1.0, kind=kind, dtype=dtype, **args_FirstDerivative
        )
        Pop = Cop * Dop
    return Pop


def PoststackLinearModelling(
    wav: npt.ArrayLike,
    nt0: int,
    spatdims: Optional[Union[int, ShapeLike]] = None,
    explicit: bool = False,
    sparse: bool = False,
    kind: str = "centered",
    name: Optional[str] = None,
) -> LinearOperator:
    r"""Post-stack linearized seismic modelling operator.

    Create operator to be applied to an elastic parameter trace (or stack of
    traces) for generation of band-limited seismic post-stack data. The input
    model and data have shape :math:`[n_{t_0} \,(\times n_x \times n_y)]`.

    Parameters
    ----------
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must have odd number of elements
        and centered to zero). If 1d, assume stationary wavelet for the entire
        time axis. If 2d, use as non-stationary wavelet (user must provide
        one wavelet per time sample in an array of size
        :math:`[n_{t_0} \times n_\text{wav}]` where :math:`n_\text{wav}` is the length
        of each wavelet). Note that the ``dtype`` of this variable will define
        that of the operator
    nt0 : :obj:`int`
        Number of samples along time axis
    spatdims : :obj:`int` or :obj:`tuple`, optional
        Number of samples along spatial axis (or axes)
        (``None`` if only one dimension is available)
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix (``True``,
        preferred for small data)
    sparse : :obj:`bool`, optional
        Create a sparse matrix (``True``) or dense  (``False``) when
        ``explicit=True``
    kind : :obj:`str`, optional
        Derivative kind (``forward`` or ``centered``).
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    Pop : :obj:`LinearOperator`
        post-stack modelling operator.

    Raises
    ------
    ValueError
        If ``wav`` is two dimensional but does not contain ``nt0`` wavelets

    Notes
    -----
    Post-stack seismic modelling is the process of constructing seismic
    post-stack data from a profile of an elastic parameter of choice in time
    (or depth) domain. This can be easily achieved using the following
    forward model:

    .. math::
        d(t, \theta=0) =  w(t) * \frac{\mathrm{d}\ln m(t)}{\mathrm{d}t}

    where :math:`m(t)` is the elastic parameter profile and
    :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{W} \mathbf{D} \mathbf{m}

    In the special case of acoustic impedance (:math:`m(t)=AI(t)`), the
    modelling operator can be used to create zero-offset data:

    .. math::
        d(t, \theta=0) = \frac{1}{2} w(t) * \frac{\mathrm{d}\ln m(t)}{\mathrm{d}t}

    where the scaling factor :math:`\frac{1}{2}` can be easily included in
    the wavelet.

    """
    Pop = _PoststackLinearModelling(
        wav, nt0, spatdims=spatdims, explicit=explicit, sparse=sparse, kind=kind
    )
    Pop.name = name
    return Pop


def PoststackInversion(
    data: NDArray,
    wav: npt.ArrayLike,
    m0: Optional[NDArray] = None,
    explicit: bool = False,
    simultaneous: bool = False,
    epsI: Optional[float] = None,
    epsR: Optional[float] = None,
    dottest: bool = False,
    epsRL1: Optional[float] = None,
    **kwargs_solver
) -> Tuple[NDArray, NDArray]:
    r"""Post-stack linearized seismic inversion.

    Invert post-stack seismic operator to retrieve an elastic parameter of
    choice from band-limited seismic post-stack data.
    Depending on the choice of input parameters, inversion can be
    trace-by-trace with explicit operator or global with either
    explicit or linear operator.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Band-limited seismic post-stack data of size
        :math:`[n_{t_0}\,(\times n_x \times n_y)]`
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must have odd number of elements
        and centered to zero). If 1d, assume stationary wavelet for the entire
        time axis. If 2d of size :math:`[n_{t_0} \times n_h]` use as
        non-stationary wavelet
    m0 : :obj:`np.ndarray`, optional
        Background model of size :math:`[n_{t_0}\,(\times n_x \times n_y)]`
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preferred for small data)
    simultaneous : :obj:`bool`, optional
        Simultaneously invert entire data (``True``) or invert
        trace-by-trace (``False``) when using ``explicit`` operator
        (note that the entire data is always inverted when working
        with linear operator)
    epsI : :obj:`float`, optional
        Damping factor for Tikhonov regularization term
    epsR : :obj:`float`, optional
        Damping factor for additional Laplacian regularization term
    dottest : :obj:`bool`, optional
        Apply dot-test
    epsRL1 : :obj:`float`, optional
        Damping factor for additional blockiness regularization term
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.linalg.lstsq`
        solver (if ``explicit=True`` and  ``epsR=None``)
        or :py:func:`scipy.sparse.linalg.lsqr` solver (if ``explicit=False``
        and/or ``epsR`` is not ``None``)

    Returns
    -------
    minv : :obj:`np.ndarray`
        Inverted model of size :math:`[n_{t_0}\,(\times n_x \times n_y)]`
    datar : :obj:`np.ndarray`
        Residual data (i.e., data - background data) of
        size :math:`[n_{t_0}\,(\times n_x \times n_y)]`

    Notes
    -----
    The cost function and solver used in the seismic post-stack inversion
    module depends on the choice of ``explicit``, ``simultaneous``, ``epsI``,
    and ``epsR`` parameters:

    * ``explicit=True``, ``epsI=None`` and ``epsR=None``: the explicit
      solver :py:func:`scipy.linalg.lstsq` is used if ``simultaneous=False``
      (or the iterative solver :py:func:`scipy.sparse.linalg.lsqr` is used
      if ``simultaneous=True``)
    * ``explicit=True`` with ``epsI`` and ``epsR=None``: the regularized
      normal equations :math:`\mathbf{W}^T\mathbf{d} = (\mathbf{W}^T
      \mathbf{W} + \epsilon_\mathbf{I}^2 \mathbf{I}) \mathbf{AI}` are instead fed
      into the :py:func:`scipy.linalg.lstsq` solver if ``simultaneous=False``
      (or the iterative solver :py:func:`scipy.sparse.linalg.lsqr`
      if ``simultaneous=True``)
    * ``explicit=False`` and ``epsR=None``: the iterative solver
      :py:func:`scipy.sparse.linalg.lsqr` is used
    * ``explicit=False`` with ``epsR`` and ``epsRL1=None``: the iterative
      solver :py:func:`pylops.optimization.leastsquares.regularized_inversion`
      is used to solve the spatially regularized problem.
    * ``explicit=False`` with ``epsR`` and ``epsRL1``: the iterative
      solver :py:func:`pylops.optimization.sparsity.SplitBregman`
      is used to solve the blockiness-promoting (in vertical direction)
      and spatially regularized (in additional horizontal directions) problem.

    Note that the convergence of iterative solvers such as
    :py:func:`scipy.sparse.linalg.lsqr` can be very slow for this type of
    operator. It is suggested to take a two steps approach with first a
    trace-by-trace inversion using the explicit operator, followed by a
    regularized global inversion using the outcome of the previous
    inversion as initial guess.
    """
    ncp = get_array_module(wav)

    # check if background model and data have same shape
    if m0 is not None and data.shape != m0.shape:
        raise ValueError("data and m0 must have same shape")

    nspat: Optional[Union[int, ShapeLike]]
    # find out dimensions
    if data.ndim == 1:
        dims = 1
        nt0 = data.size
        nspat = None
        nspatprod = nx = 1
    elif data.ndim == 2:
        dims = 2
        nt0, nx = data.shape
        nspat = (nx,)
        nspatprod = nx
    else:
        dims = 3
        nt0, nx, ny = data.shape
        nspat = (nx, ny)
        nspatprod = nx * ny
        data = data.reshape(nt0, nspatprod)

    # create operator
    PPop = PoststackLinearModelling(wav, nt0=nt0, spatdims=nspat, explicit=explicit)
    if dottest:
        Dottest(
            PPop,
            nt0 * nspatprod,
            nt0 * nspatprod,
            raiseerror=True,
            backend=get_module_name(ncp),
            verb=True,
        )

    # create and remove background data from original data
    datar = data.ravel() if m0 is None else data.ravel() - PPop * m0.ravel()
    # invert model
    if epsR is None:
        # inversion without spatial regularization
        if explicit:
            if epsI is None and not simultaneous:
                # solve unregularized equations indipendently trace-by-trace
                minv = get_lstsq(data)(
                    PPop.A, datar.reshape(nt0, nspatprod).squeeze(), **kwargs_solver
                )[0]
            elif epsI is None and simultaneous:
                # solve unregularized equations simultaneously
                if ncp == np:
                    minv = lsqr(PPop, datar, **kwargs_solver)[0]
                else:
                    minv = cgls(
                        PPop,
                        datar,
                        x0=ncp.zeros(int(PPop.shape[1]), PPop.dtype),
                        **kwargs_solver
                    )[0]
            elif epsI is not None:
                # create regularized normal equations
                PP = ncp.dot(PPop.A.T, PPop.A) + epsI * ncp.eye(nt0, dtype=PPop.A.dtype)
                datarn = ncp.dot(PPop.A.T, datar.reshape(nt0, nspatprod))
                if not simultaneous:
                    # solve regularized normal eqs. trace-by-trace
                    minv = get_lstsq(data)(PP, datarn, **kwargs_solver)[0]
                else:
                    # solve regularized normal equations simultaneously
                    PPop_reg = MatrixMult(PP, otherdims=nspatprod)
                    if ncp == np:
                        minv = lsqr(PPop_reg, datar.ravel(), **kwargs_solver)[0]
                    else:
                        minv = cgls(
                            PPop_reg,
                            datar.ravel(),
                            x0=ncp.zeros(int(PPop_reg.shape[1]), PPop_reg.dtype),
                            **kwargs_solver
                        )[0]
            else:
                # create regularized normal eqs. and solve them simultaneously
                PP = ncp.dot(PPop.A.T, PPop.A) + epsI * ncp.eye(nt0, dtype=PPop.A.dtype)
                datarn = PPop.A.T * datar.reshape(nt0, nspatprod)
                PPop_reg = MatrixMult(PP, otherdims=nspatprod)
                minv = get_lstsq(data)(PPop_reg.A, datarn.ravel(), **kwargs_solver)[0]
        else:
            # solve unregularized normal equations simultaneously with lop
            if ncp == np:
                minv = lsqr(PPop, datar, **kwargs_solver)[0]
            else:
                minv = cgls(
                    PPop,
                    datar,
                    x0=ncp.zeros(int(PPop.shape[1]), PPop.dtype),
                    **kwargs_solver
                )[0]
    else:
        if epsRL1 is None:
            # L2 inversion with spatial regularization
            if dims == 1:
                Regop = SecondDerivative(nt0, dtype=PPop.dtype)
            elif dims == 2:
                Regop = Laplacian((nt0, nx), dtype=PPop.dtype)
            else:
                Regop = Laplacian((nt0, nx, ny), axes=(1, 2), dtype=PPop.dtype)

            minv = regularized_inversion(
                PPop,
                data.ravel(),
                [Regop],
                x0=None if m0 is None else m0.ravel(),
                epsRs=[epsR],
                **kwargs_solver
            )[0]
        else:
            # Blockiness-promoting inversion with spatial regularization
            if dims == 1:
                RegL1op = FirstDerivative(nt0, kind="forward", dtype=PPop.dtype)
                RegL2op = None
            elif dims == 2:
                RegL1op = FirstDerivative(
                    (nt0, nx), axis=0, kind="forward", dtype=PPop.dtype
                )
                RegL2op = SecondDerivative((nt0, nx), axis=1, dtype=PPop.dtype)
            else:
                RegL1op = FirstDerivative(
                    (nt0, nx, ny), axis=0, kind="forward", dtype=PPop.dtype
                )
                RegL2op = Laplacian((nt0, nx, ny), axes=(1, 2), dtype=PPop.dtype)

            if "mu" in kwargs_solver.keys():
                mu = kwargs_solver["mu"]
                kwargs_solver.pop("mu")
            else:
                mu = 1.0
            if "niter_outer" in kwargs_solver.keys():
                niter_outer = kwargs_solver["niter_outer"]
                kwargs_solver.pop("niter_outer")
            else:
                niter_outer = 3
            if "niter_inner" in kwargs_solver.keys():
                niter_inner = kwargs_solver["niter_inner"]
                kwargs_solver.pop("niter_inner")
            else:
                niter_inner = 5
            if not isinstance(epsRL1, (list, tuple)):
                epsRL1 = list([epsRL1])
            if not isinstance(epsR, (list, tuple)):
                epsR = list([epsR])
            minv = splitbregman(
                PPop,
                data.ravel(),
                [RegL1op],
                RegsL2=[RegL2op],
                epsRL1s=epsRL1,
                epsRL2s=epsR,
                mu=mu,
                niter_outer=niter_outer,
                niter_inner=niter_inner,
                x0=None if m0 is None else m0.ravel(),
                **kwargs_solver
            )[0]

    # compute residual
    if epsR is None:
        datar -= PPop * minv.ravel()
    else:
        datar = data.ravel() - PPop * minv.ravel()

    # reshape inverted model and residual data
    if dims == 1:
        minv = minv.squeeze()
        datar = datar.squeeze()
    elif dims == 2:
        minv = minv.reshape(nt0, nx)
        datar = datar.reshape(nt0, nx)
    else:
        minv = minv.reshape(nt0, nx, ny)
        datar = datar.reshape(nt0, nx, ny)

    if m0 is not None and epsR is None:
        minv = minv + m0

    return minv, datar
