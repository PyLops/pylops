import logging

import numpy as np

from pylops import Laplacian, Restriction, SecondDerivative
from pylops.optimization.leastsquares import RegularizedInversion
from pylops.optimization.sparsity import FISTA
from pylops.signalprocessing import (
    FFT2D,
    FFTND,
    ChirpRadon2D,
    ChirpRadon3D,
    Interp,
    Radon2D,
    Radon3D,
    Sliding2D,
    Sliding3D,
)
from pylops.utils.backend import get_array_module
from pylops.utils.dottest import dottest as Dottest

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def SeismicInterpolation(
    data,
    nrec,
    iava,
    iava1=None,
    kind="fk",
    nffts=None,
    sampling=None,
    spataxis=None,
    spat1axis=None,
    taxis=None,
    paxis=None,
    p1axis=None,
    centeredh=True,
    nwins=None,
    nwin=None,
    nover=None,
    design=False,
    engine="numba",
    dottest=False,
    **kwargs_solver
):
    r"""Seismic interpolation (or regularization).

    Interpolate seismic data from irregular to regular spatial grid.
    Depending on the size of the input ``data``, interpolation is either
    2- or 3-dimensional. In case of 3-dimensional interpolation,
    data can be irregularly sampled in either one or both spatial directions.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Irregularly sampled seismic data of size
        :math:`[n_{r_y} \,(\times n_{r_x} \times n_t)]`
    nrec : :obj:`int` or :obj:`tuple`
        Number of elements in the regularly sampled (reconstructed) spatial
        array, :math:`n_{R_y}` for 2-dimensional data and
        :math:`(n_{R_y}, n_{R_x})` for 3-dimensional data
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer (or floating) indices of locations of available samples in
        first dimension of regularly sampled spatial grid of interpolated
        signal. The :class:`pylops.basicoperators.Restriction` operator is
        used in case of integer indices, while the
        :class:`pylops.signalprocessing.Iterp` operator is used in
        case of floating indices.
    iava1 : :obj:`list` or :obj:`numpy.ndarray`, optional
        Integer (or floating) indices of locations of available samples in
        second dimension of regularly sampled spatial grid of interpolated
        signal. Can be used only in case of 3-dimensional data.
    kind : :obj:`str`, optional
        Type of inversion: ``fk`` (default), ``spatial``, ``radon-linear``,
        ``chirpradon-linear``, ``radon-parabolic`` , ``radon-hyperbolic``,
        ``sliding``, or ``chirp-sliding``
    nffts : :obj:`int` or :obj:`tuple`, optional
        nffts : :obj:`tuple`, optional
        Number of samples in Fourier Transform for each direction.
        Required if ``kind='fk'``
    sampling : :obj:`tuple`, optional
        Sampling steps ``dy`` (, ``dx``) and ``dt``. Required if ``kind='fk'``
        or ``kind='radon-linear'``
    spataxis : :obj:`np.ndarray`, optional
        First spatial axis. Required for ``kind='radon-linear'``,
        ``kind='chirpradon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'``, can also be provided instead of
        ``sampling`` for ``kind='fk'``
    spat1axis : :obj:`np.ndarray`, optional
        Second spatial axis. Required for ``kind='radon-linear'``,
        ``kind='chirpradon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'``, can also be provided instead of
        ``sampling`` for ``kind='fk'``
    taxis : :obj:`np.ndarray`, optional
        Time axis. Required for ``kind='radon-linear'``,
        ``kind='chirpradon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'``, can also be provided instead of
        ``sampling`` for ``kind='fk'``
    paxis : :obj:`np.ndarray`, optional
        First Radon axis. Required for ``kind='radon-linear'``,
        ``kind='chirpradon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'``, ``kind='sliding'``, and
        ``kind='chirp-sliding'``
    p1axis : :obj:`np.ndarray`, optional
        Second Radon axis. Required for ``kind='radon-linear'``,
        ``kind='chirpradon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'``, ``kind='sliding'``, and
        ``kind='chirp-sliding'``
    centeredh : :obj:`bool`, optional
        Assume centered spatial axis (``True``) or not (``False``).
        Required for ``kind='radon-linear'``, ``kind='radon-parabolic'``
        and ``kind='radon-hyperbolic'``
    nwins : :obj:`int` or :obj:`tuple`, optional
        Number of windows. Required for ``kind='sliding'`` and
        ``kind='chirp-sliding'``
    nwin : :obj:`int` or :obj:`tuple`, optional
        Number of samples of window. Required for ``kind='sliding'`` and
        ``kind='chirp-sliding'``
    nover : :obj:`int` or :obj:`tuple`, optional
        Number of samples of overlapping part of window. Required for
        ``kind='sliding'`` and ``kind='chirp-sliding'``
    design : :obj:`bool`, optional
        Print number of sliding window (``True``) or not (``False``) when
        using ``kind='sliding'`` and ``kind='chirp-sliding'``
    engine : :obj:`str`, optional
        Engine used for Radon computations (``numpy/numba``
        for ``Radon2D`` and ``Radon3D`` or ``numpy/fftw``
        for ``ChirpRadon2D`` and ``ChirpRadon3D``)
    dottest : :obj:`bool`, optional
        Apply dot-test
    **kwargs_solver
        Arbitrary keyword arguments for
        :py:func:`pylops.optimization.leastsquares.RegularizedInversion` solver
        if ``kind='spatial'`` or
        :py:func:`pylops.optimization.sparsity.FISTA` solver otherwise

    Returns
    -------
    recdata : :obj:`np.ndarray`
        Reconstructed data of size :math:`[n_{R_y}\,(\times n_{R_x} \times n_t)]`
    recprec : :obj:`np.ndarray`
        Reconstructed data in the sparse or preconditioned domain in case of
        ``kind='fk'``, ``kind='radon-linear'``, ``kind='radon-parabolic'``,
        ``kind='radon-hyperbolic'`` and ``kind='sliding'``
    cost : :obj:`np.ndarray`
        Cost function norm

    Raises
    ------
    KeyError
        If ``kind`` is neither ``spatial``, ``fl``, ``radon-linear``,
        ``radon-parabolic``, ``radon-hyperbolic`` nor ``sliding``

    Notes
    -----
    The problem of seismic data interpolation (or regularization) can be
    formally written as

    .. math::
        \mathbf{y} = \mathbf{R} \mathbf{x}

    where a restriction or interpolation operator is applied along the spatial
    direction(s). Here :math:`\mathbf{y} = [\mathbf{y}_{R1}^T, \mathbf{y}_{R2}^T,\ldots,
    \mathbf{y}_{RN^T}]^T` where each vector :math:`\mathbf{y}_{Ri}`
    contains all time samples recorded in the seismic data at the specific
    receiver :math:`R_i`. Similarly, :math:`\mathbf{x} = [\mathbf{x}_{r1}^T,
    \mathbf{x}_{r2}^T,\ldots, \mathbf{x}_{rM}^T]`, contains all traces at the
    regularly and finely sampled receiver locations :math:`r_i`.

    Several alternative approaches can be taken to solve such a problem. They
    mostly differ in the choice of the regularization (or preconditining) used
    to mitigate the ill-posedness of the problem:

        * ``spatial``: least-squares inversion in the original time-space domain
          with an additional spatial smoothing regularization term,
          corresponding to the cost function
          :math:`J = ||\mathbf{y} - \mathbf{R} \mathbf{x}||_2 +
          \epsilon_\nabla \nabla ||\mathbf{x}||_2` where :math:`\nabla` is
          a second order space derivative implemented via
          :class:`pylops.basicoperators.SecondDerivative` in 2-dimensional case
          and :class:`pylops.basicoperators.Laplacian` in 3-dimensional case
        * ``fk``: L1 inversion in frequency-wavenumber preconditioned domain
          corresponding to the cost function
          :math:`J = ||\mathbf{y} - \mathbf{R} \mathbf{F} \mathbf{x}||_2` where
          :math:`\mathbf{F}` is frequency-wavenumber transform implemented via
          :class:`pylops.signalprocessing.FFT2D` in 2-dimensional case
          and :class:`pylops.signalprocessing.FFTND` in 3-dimensional case
        * ``radon-linear``: L1 inversion in linear Radon preconditioned domain
          using the same cost function as ``fk`` but with :math:`\mathbf{F}`
          being a Radon transform implemented via
          :class:`pylops.signalprocessing.Radon2D` in 2-dimensional case
          and :class:`pylops.signalprocessing.Radon3D` in 3-dimensional case
        * ``radon-parabolic``: L1 inversion in parabolic Radon
          preconditioned domain
        * ``radon-hyperbolic``: L1 inversion in hyperbolic Radon
          preconditioned domain
        * ``sliding``: L1 inversion in sliding-linear Radon
          preconditioned domain using the same cost function as ``fk``
          but with :math:`\mathbf{F}` being a sliding Radon transform
          implemented via :class:`pylops.signalprocessing.Sliding2D` in
          2-dimensional case and :class:`pylops.signalprocessing.Sliding3D`
          in 3-dimensional case

    """
    ncp = get_array_module(data)

    dtype = data.dtype
    ndims = data.ndim
    if ndims == 1 or ndims > 3:
        raise ValueError("data must have 2 or 3 dimensions")
    if ndims == 2:
        dimsd = data.shape
        dims = (nrec, dimsd[1])
    else:
        dimsd = data.shape
        dims = (nrec[0], nrec[1], dimsd[2])

    # sampling
    if taxis is not None:
        dt = taxis[1] - taxis[0]
    if spataxis is not None:
        dspat = np.abs(spataxis[1] - spataxis[0])
    if spat1axis is not None:
        dspat1 = np.abs(spat1axis[1] - spat1axis[0])

    # create restriction/interpolation operator
    if iava.dtype == float:
        Rop = Interp(np.prod(dims), iava, dims=dims, dir=0, kind="linear", dtype=dtype)
        if ndims == 3 and iava1 is not None:
            dims1 = (len(iava), nrec[1], dimsd[2])
            Rop1 = Interp(
                np.prod(dims1), iava1, dims=dims1, dir=1, kind="linear", dtype=dtype
            )
            Rop = Rop1 * Rop
    else:
        Rop = Restriction(np.prod(dims), iava, dims=dims, dir=0, dtype=dtype)
        if ndims == 3 and iava1 is not None:
            dims1 = (len(iava), nrec[1], dimsd[2])
            Rop1 = Restriction(np.prod(dims1), iava1, dims=dims1, dir=1, dtype=dtype)
            Rop = Rop1 * Rop

    # create other operators for inversion
    if kind == "spatial":
        prec = False
        dotcflag = 0
        if ndims == 3 and iava1 is not None:
            Regop = Laplacian(dims=dims, dirs=(0, 1), dtype=dtype)
        else:
            Regop = SecondDerivative(np.prod(dims), dims=(dims), dir=0, dtype=dtype)
        SIop = Rop
    elif kind == "fk":
        prec = True
        dimsp = nffts
        dotcflag = 1
        if ndims == 3:
            if sampling is None:
                if spataxis is None or spat1axis is None or taxis is None:
                    raise ValueError(
                        "Provide either sampling or spataxis, "
                        "spat1axis and taxis for kind=%s" % kind
                    )
                else:
                    sampling = (
                        np.abs(spataxis[1] - spataxis[1]),
                        np.abs(spat1axis[1] - spat1axis[1]),
                        np.abs(taxis[1] - taxis[1]),
                    )
            Pop = FFTND(dims=dims, nffts=nffts, sampling=sampling)
            Pop = Pop.H
        else:
            if sampling is None:
                if spataxis is None or taxis is None:
                    raise ValueError(
                        "Provide either sampling or spataxis, "
                        "and taxis for kind=%s" % kind
                    )
                else:
                    sampling = (
                        np.abs(spataxis[1] - spataxis[1]),
                        np.abs(taxis[1] - taxis[1]),
                    )
            Pop = FFT2D(dims=dims, nffts=nffts, sampling=sampling)
            Pop = Pop.H
        SIop = Rop * Pop
    elif "chirpradon" in kind:
        prec = True
        dotcflag = 0
        if ndims == 3:
            Pop = ChirpRadon3D(
                taxis,
                spataxis,
                spat1axis,
                (np.max(paxis) * dspat / dt, np.max(p1axis) * dspat1 / dt),
            ).H
            dimsp = (spataxis.size, spat1axis.size, taxis.size)
        else:
            Pop = ChirpRadon2D(taxis, spataxis, np.max(paxis) * dspat / dt).H
            dimsp = (spataxis.size, taxis.size)
        SIop = Rop * Pop
    elif "radon" in kind:
        prec = True
        dotcflag = 0
        kindradon = kind.split("-")[-1]
        if ndims == 3:
            Pop = Radon3D(
                taxis,
                spataxis,
                spat1axis,
                paxis,
                p1axis,
                centeredh=centeredh,
                kind=kindradon,
                engine=engine,
            )
            dimsp = (paxis.size, p1axis.size, taxis.size)

        else:
            Pop = Radon2D(
                taxis,
                spataxis,
                paxis,
                centeredh=centeredh,
                kind=kindradon,
                engine=engine,
            )
            dimsp = (paxis.size, taxis.size)
        SIop = Rop * Pop
    elif kind in ("sliding", "chirp-sliding"):
        prec = True
        dotcflag = 0
        if ndims == 3:
            nspat, nspat1 = spataxis.size, spat1axis.size
            spataxis_local = np.linspace(
                -dspat * nwin[0] // 2, dspat * nwin[0] // 2, nwin[0]
            )
            spat1axis_local = np.linspace(
                -dspat1 * nwin[1] // 2, dspat1 * nwin[1] // 2, nwin[1]
            )
            dimsslid = (nspat, nspat1, taxis.size)
            if kind == "sliding":
                npaxis, np1axis = paxis.size, p1axis.size
                Op = Radon3D(
                    taxis,
                    spataxis_local,
                    spat1axis_local,
                    paxis,
                    p1axis,
                    centeredh=True,
                    kind="linear",
                    engine=engine,
                )
            else:
                npaxis, np1axis = nwin[0], nwin[1]
                Op = ChirpRadon3D(
                    taxis,
                    spataxis_local,
                    spat1axis_local,
                    (np.max(paxis) * dspat / dt, np.max(p1axis) * dspat1 / dt),
                ).H
            dimsp = (nwins[0] * npaxis, nwins[1] * np1axis, dimsslid[2])
            Pop = Sliding3D(
                Op, dimsp, dimsslid, nwin, nover, (npaxis, np1axis), tapertype="cosine"
            )
            # to be able to reshape correctly the preconditioned model
            dimsp = (nwins[0], nwins[1], npaxis, np1axis, dimsslid[2])
        else:
            nspat = spataxis.size
            spataxis_local = np.linspace(-dspat * nwin // 2, dspat * nwin // 2, nwin)
            dimsslid = (nspat, taxis.size)
            if kind == "sliding":
                npaxis = paxis.size
                Op = Radon2D(
                    taxis,
                    spataxis_local,
                    paxis,
                    centeredh=True,
                    kind="linear",
                    engine=engine,
                )
            else:
                npaxis = nwin
                Op = ChirpRadon2D(taxis, spataxis_local, np.max(paxis) * dspat / dt).H
            dimsp = (nwins * npaxis, dimsslid[1])
            Pop = Sliding2D(
                Op, dimsp, dimsslid, nwin, nover, tapertype="cosine", design=design
            )
        SIop = Rop * Pop
    else:
        raise KeyError(
            "kind must be spatial, fk, radon-linear, "
            "radon-parabolic, radon-hyperbolic, sliding or chirp-sliding"
        )

    # dot-test
    if dottest:
        Dottest(
            SIop,
            np.prod(dimsd),
            np.prod(dimsp) if prec else np.prod(dims),
            complexflag=dotcflag,
            raiseerror=True,
            verb=True,
        )

    # inversion
    if kind == "spatial":
        recdata = RegularizedInversion(SIop, [Regop], data.ravel(), **kwargs_solver)
        if isinstance(recdata, tuple):
            recdata = recdata[0]
        recdata = recdata.reshape(dims)
        recprec = None
        cost = None
    else:
        recprec = FISTA(SIop, data.ravel(), **kwargs_solver)
        if len(recprec) == 3:
            cost = recprec[2]
        else:
            cost = None
        recprec = recprec[0]
        recdata = np.real(Pop * recprec)

        recprec = recprec.reshape(dimsp)
        recdata = recdata.reshape(dims)

    return recdata, recprec, cost
