import logging
import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr

from pylops.signalprocessing import Convolve1D
from pylops.utils.signalprocessing import convmtx
from pylops.utils import dottest as Dottest
from pylops import MatrixMult, FirstDerivative, SecondDerivative, Laplacian
from pylops.optimization.leastsquares import RegularizedInversion

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def PoststackLinearModelling(wav, nt0, spatdims=None, explicit=False):
    r"""Post-stack linearized seismic modelling operator.

    Create operator to be applied to acoustic impedance profile
    for generation of band-limited seismic post-stack data.

    Parameters
    ----------
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of elements
        and centered to zero)
    nt0 : :obj:`int`
        Number of samples along time axis
    spatdims : :obj:`int` or :obj:`tuple`, optional
        Number of samples along spatial axis (or axes)
        (``None`` if only one dimension is available)
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix (``True``,
        preferred for small data)

    Returns
    -------
    Postop : :obj:`LinearOperator`
        post-stack modelling operator.

    Notes
    -----
    Post-stack seismic modelling is the process of constructing
    seismic post-stack data from a profile of acoustic impedance in
    time (or depth) domain. This can be easily achieved using the
    following forward model:

    .. math::
        d(t, \theta=0) = w(t) * \frac{dln(AI(t))}{dt}

    where :math:`AI(t)` is the acoustic impedance profile and
    :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{W} \mathbf{AI}

    On the other hand, post-stack inversion aims at recovering
    the impedance profile from the band-limited seismic stack data.

    """
    # organize dimensions
    if spatdims is None:
        dims = (nt0, )
        spatdims = None
        #spatdims = (1, )
    elif isinstance(spatdims, int):
        dims = (nt0, spatdims)
        spatdims = (spatdims,)
    else:
        dims = (nt0, ) + spatdims

    if explicit:
        # Create derivative operator
        D = np.diag(0.5 * np.ones(nt0 - 1), k=1) - \
            np.diag(0.5 * np.ones(nt0 - 1), -1)
        D[0] = D[-1] = 0

        # Create wavelet operator
        C = convmtx(wav, nt0)[:, len(wav) // 2:-len(wav) // 2 + 1]

        # Combine operators
        M = np.dot(C, D)
        return MatrixMult(M, dims=spatdims)
    else:
        # Create wavelet operator
        Cop = Convolve1D(np.prod(np.array(dims)), h=wav,
                         offset=len(wav)//2, dir=0, dims=dims)

        # Create derivative operator
        Dop = FirstDerivative(np.prod(np.array(dims)), dims=dims,
                              dir=0, sampling=1.)

        return Cop*Dop


def PoststackInversion(data, wav, m0=None, explicit=False, simultaneous=False,
                       epsI=None, epsR=None, dottest=False,
                       **kwargs_solver):
    r"""Post-stack linearized seismic inversion.

    Invert post-stack seismic operator to retrieve an acoustic
    impedance profile from band-limited seismic post-stack data.
    Depending on the choice of input parameters, inversion can be
    trace-by-trace with explicit operator or global with either
    explicit or linear operator.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Band-limited seismic post-stack data of size
        :math:`[n_{t0} (\times n_x \times n_y)]`
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of elements
        and centered to zero)
    m0 : :obj:`np.ndarray`, optional
        Background model of size :math:`[n_{t0} (\times n_x \times n_y)]`
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
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.linalg.lstsq`
        solver (if ``explicit=True`` and  ``epsR=None``)
        or :py:func:`scipy.sparse.linalg.lsqr` solver (if ``explicit=False``
        and/or ``epsR`` is not ``None``))

    Returns
    -------
    minv : :obj:`np.ndarray`
        Inverted model of size :math:`[n_{t0} (\times n_x \times n_y)]`
    datar : :obj:`np.ndarray`
        Residual data (i.e., data - background data) of
        size :math:`[n_{t0} (\times n_x \times n_y)]`

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
      \mathbf{W} + \epsilon_I^2 \mathbf{I}) \mathbf{AI}` are instead fed
      into the :py:func:`scipy.linalg.lstsq` solver if ``simultaneous=False``
      (or the iterative solver :py:func:`scipy.sparse.linalg.lsqr`
      if ``simultaneous=True``)
    * ``explicit=False`` and ``epsR=None``: the iterative solver
      :py:func:`scipy.sparse.linalg.lsqr` is used
    * ``explicit=False`` with ``epsR``: the iterative solver
      :py:func:`pylops.optimization.leastsquares.RegularizedInversion` is used
      to solve the spatially regularized problem.

    Note that the convergence of iterative solvers such as
    :py:func:`scipy.sparse.linalg.lsqr` can be very slow for this type of
    operator. It is suggested to take a two steps approach with first a
    trace-by-trace inversion using the explicit operator, followed by a
    regularized global inversion using the outcome of the previous
    inversion as initial guess.
    """
    # check if background model and data have same shape
    if m0 is not None and data.shape != m0.shape:
        print(m0.shape, data.shape)
        raise ValueError('data and m0 must have same shape')

    # find out dimensions
    if data.ndim == 1:
        dims = 1
        nt0 = data.size
        nspat = None
        nspatprod = nx = 1
    elif data.ndim == 2:
        dims = 2
        nt0, nx = data.shape
        nspat = (nx, )
        nspatprod = nx
    else:
        dims = 3
        nt0, nx, ny = data.shape
        nspat = (nx, ny)
        nspatprod = nx*ny
        data = data.reshape(nt0, nspatprod)

    # create operator
    PPop = PoststackLinearModelling(wav, nt0=nt0,
                                    spatdims=nspat, explicit=explicit)
    if dottest:
        Dottest(PPop, nt0*nspatprod, nt0*nspatprod, raiseerror=True, verb=True)

    # create and remove background data from original data
    datar = data.flatten() if m0 is None else \
        data.flatten() - PPop * m0.flatten()
    # invert model
    if epsR is None:
        # inversion without spatial regularization
        if explicit:
            if epsI is None and not simultaneous:
                # solve unregularized equations indipendently trace-by-trace
                minv = lstsq(PPop.A, datar.reshape(nt0, nspatprod).squeeze(),
                             **kwargs_solver)[0]
            elif epsI is None and simultaneous:
                # solve unregularized equations simultaneously
                minv = lsqr(PPop, datar, **kwargs_solver)[0]
            elif epsI is not None:
                # create regularized normal equations
                PP = np.dot(PPop.A.T, PPop.A) + epsI * np.eye(nt0)
                datarn = np.dot(PPop.A.T, datar.reshape(nt0, nspatprod))
                if not simultaneous:
                    # solve regularized normal eqs. trace-by-trace
                    minv = lstsq(PP, datarn,
                                 **kwargs_solver)[0]
                else:
                    # solve regularized normal equations simultaneously
                    PPop_reg = MatrixMult(PP, dims=nspatprod)
                    minv = lsqr(PPop_reg, datar.flatten(), **kwargs_solver)[0]
            else:
                # create regularized normal eqs. and solve them simultaneously
                PP = np.dot(PPop.A.T, PPop.A) + epsI * np.eye(nt0)
                datarn = PPop.A.T * datar.reshape(nt0, nspatprod)
                PPop_reg = MatrixMult(PP, dims=nspatprod)
                minv = lstsq(PPop_reg, datarn.flatten(), **kwargs_solver)[0]
        else:
            # solve unregularized normal equations simultaneously with lop
            minv = lsqr(PPop, datar, **kwargs_solver)[0]
    else:
        # inversion with spatial regularization
        if dims==1:
            Regop = SecondDerivative(nt0, dtype=PPop.dtype)
        elif dims==2:
            Regop = Laplacian((nt0, nx), dtype=PPop.dtype)
        else:
            Regop = Laplacian((nt0, nx, ny), dirs=(1, 2), dtype=PPop.dtype)

        minv = RegularizedInversion(PPop, [Regop], data.flatten(),
                                    x0=None if m0 is None else m0.flatten(),
                                    epsRs=[epsR], returninfo=False,
                                    **kwargs_solver)

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
