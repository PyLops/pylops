import logging

import numpy as np
from scipy.sparse.linalg import lsqr

from pylops import (
    Diagonal,
    FirstDerivative,
    Identity,
    Laplacian,
    MatrixMult,
    SecondDerivative,
    VStack,
)
from pylops.avo.avo import AVOLinearModelling, akirichards, fatti, ps
from pylops.optimization.leastsquares import RegularizedInversion
from pylops.optimization.solver import cgls
from pylops.optimization.sparsity import SplitBregman
from pylops.signalprocessing import Convolve1D
from pylops.utils import dottest as Dottest
from pylops.utils.backend import (
    get_array_module,
    get_block_diag,
    get_lstsq,
    get_module_name,
)
from pylops.utils.signalprocessing import convmtx

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

_linearizations = {"akirich": 3, "fatti": 3, "ps": 3}


def PrestackLinearModelling(
    wav,
    theta,
    vsvp=0.5,
    nt0=1,
    spatdims=None,
    linearization="akirich",
    explicit=False,
    kind="centered",
):
    r"""Pre-stack linearized seismic modelling operator.

    Create operator to be applied to elastic property profiles
    for generation of band-limited seismic angle gathers from a
    linearized version of the Zoeppritz equation. The input model must
    be arranged in a vector of size :math:`n_m \times n_{t_0}\,(\times n_x \times n_y)`
    for ``explicit=True`` and :math:`n_{t_0} \times n_m \,(\times n_x \times n_y)`
    for ``explicit=False``. Similarly the output data is arranged in a
    vector of size :math:`n_{\theta} \times n_{t_0} \,(\times n_x \times n_y)`
    for ``explicit=True`` and :math:`n_{t_0} \times n_{\theta} \,(\times n_x \times n_y)`
    for ``explicit=False``.

    Parameters
    ----------
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of
        elements and centered to zero). Note that the ``dtype`` of this
        variable will define that of the operator
    theta : :obj:`np.ndarray`
        Incident angles in degrees. Must have same ``dtype`` of ``wav`` (or
        it will be automatically casted to it)
    vsvp : :obj:`float` or :obj:`np.ndarray`
        :math:`V_S/V_P` ratio (constant or time/depth variant)
    nt0 : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalar)
    spatdims : :obj:`int` or :obj:`tuple`, optional
        Number of samples along spatial axis (or axes)
        (``None`` if only one dimension is available)
    linearization : `{"akirich", "fatti", "PS"}` or :obj:`callable`, optional
        * "akirich": Aki-Richards. See :py:func:`pylops.avo.avo.akirichards`.

        * "fatti": Fatti. See :py:func:`pylops.avo.avo.fatti`.

        * "PS": PS. See :py:func:`pylops.avo.avo.ps`.

        * Function with the same signature as :py:func:`pylops.avo.avo.akirichards`
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preferred for small data)
    kind : :obj:`str`, optional
        Derivative kind (``forward`` or ``centered``).

    Returns
    -------
    Preop : :obj:`LinearOperator`
        pre-stack modelling operator.

    Raises
    ------
    NotImplementedError
        If ``linearization`` is not an implemented linearization
    NotImplementedError
        If ``kind`` is not ``forward`` nor ``centered``

    Notes
    -----
    Pre-stack seismic modelling is the process of constructing seismic
    pre-stack data from three (or two) profiles of elastic parameters in time
    (or depth) domain. This can be easily achieved using the following
    forward model:

    .. math::
        d(t, \theta) = w(t) * \sum_{i=1}^{n_m} G_i(t, \theta) m_i(t)

    where :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{G} \mathbf{m}

    On the other hand, pre-stack inversion aims at recovering the different
    profiles of elastic properties from the band-limited seismic
    pre-stack data.

    """
    ncp = get_array_module(wav)

    # check kind is correctly selected
    if kind not in ["forward", "centered"]:
        raise NotImplementedError("%s not an available derivative kind..." % kind)
    # define dtype to be used
    dtype = theta.dtype  # ensure theta.dtype rules that of operator
    theta = theta.astype(dtype)

    # create vsvp profile
    vsvp = vsvp if isinstance(vsvp, ncp.ndarray) else vsvp * ncp.ones(nt0, dtype=dtype)
    nt0 = len(vsvp)
    ntheta = len(theta)

    # organize dimensions
    if spatdims is None:
        dims = (nt0, ntheta)
        spatdims = None
    elif isinstance(spatdims, int):
        dims = (nt0, ntheta, spatdims)
        spatdims = (spatdims,)
    else:
        dims = (nt0, ntheta) + spatdims

    if explicit:
        # Create AVO operator
        if linearization == "akirich":
            G = akirichards(theta, vsvp, n=nt0)
        elif linearization == "fatti":
            G = fatti(theta, vsvp, n=nt0)
        elif linearization == "ps":
            G = ps(theta, vsvp, n=nt0)
        elif callable(linearization):
            G = linearization(theta, vsvp, n=nt0)
        else:
            logging.error("%s not an available linearization...", linearization)
            raise NotImplementedError(
                "%s not an available linearization..." % linearization
            )
        nG = len(G)
        G = [
            ncp.hstack([ncp.diag(G_[itheta] * ncp.ones(nt0, dtype=dtype)) for G_ in G])
            for itheta in range(ntheta)
        ]
        G = ncp.vstack(G).reshape(ntheta * nt0, nG * nt0)

        # Create derivative operator
        if kind == "centered":
            D = ncp.diag(0.5 * ncp.ones(nt0 - 1, dtype=dtype), k=1) - ncp.diag(
                0.5 * ncp.ones(nt0 - 1, dtype=dtype), k=-1
            )
            D[0] = D[-1] = 0
        else:
            D = ncp.diag(ncp.ones(nt0 - 1, dtype=dtype), k=1) - ncp.diag(
                ncp.ones(nt0, dtype=dtype), k=0
            )
            D[-1] = 0
        D = get_block_diag(theta)(*([D] * nG))

        # Create wavelet operator
        C = convmtx(wav, nt0)[:, len(wav) // 2 : -len(wav) // 2 + 1]
        C = [C] * ntheta
        C = get_block_diag(theta)(*C)

        # Combine operators
        M = ncp.dot(C, ncp.dot(G, D))
        Preop = MatrixMult(M, dims=spatdims, dtype=dtype)

    else:
        # Create wavelet operator
        Cop = Convolve1D(
            np.prod(np.array(dims)),
            h=wav,
            offset=len(wav) // 2,
            dir=0,
            dims=dims,
            dtype=dtype,
        )

        # create AVO operator
        AVOop = AVOLinearModelling(
            theta, vsvp, spatdims=spatdims, linearization=linearization, dtype=dtype
        )

        # Create derivative operator
        dimsm = list(dims)
        dimsm[1] = AVOop.npars
        Dop = FirstDerivative(
            np.prod(np.array(dimsm)),
            dims=dimsm,
            dir=0,
            sampling=1.0,
            kind=kind,
            dtype=dtype,
        )
        Preop = Cop * AVOop * Dop
    return Preop


def PrestackWaveletModelling(
    m, theta, nwav, wavc=None, vsvp=0.5, linearization="akirich"
):
    r"""Pre-stack linearized seismic modelling operator for wavelet.

    Create operator to be applied to a wavelet for generation of
    band-limited seismic angle gathers using a linearized version
    of the Zoeppritz equation.

    Parameters
    ----------
    m : :obj:`np.ndarray`
        elastic parameter profles of size :math:`[n_{t_0} \times N]`
        where :math:`N=3,\,2`. Note that the ``dtype`` of this
        variable will define that of the operator
    theta : :obj:`int`
        Incident angles in degrees. Must have same ``dtype`` of ``m`` (or
        it will be automatically cast to it)
    nwav : :obj:`np.ndarray`
        Number of samples of wavelet to be applied/estimated
    wavc : :obj:`int`, optional
        Index of the center of the wavelet
    vsvp : :obj:`np.ndarray` or :obj:`float`, optional
        :math:`V_S/V_P` ratio
    linearization : `{"akirich", "fatti", "PS"}` or :obj:`callable`, optional
        * "akirich": Aki-Richards. See :py:func:`pylops.avo.avo.akirichards`.

        * "fatti": Fatti. See :py:func:`pylops.avo.avo.fatti`.

        * "PS": PS. See :py:func:`pylops.avo.avo.ps`.

        * Function with the same signature as :py:func:`pylops.avo.avo.akirichards`

    Returns
    -------
    Mconv : :obj:`LinearOperator`
        pre-stack modelling operator for wavelet estimation.

    Raises
    ------
    NotImplementedError
        If ``linearization`` is not an implemented linearization

    Notes
    -----
    Pre-stack seismic modelling for wavelet estimate is the process
    of constructing seismic reflectivities using three (or two)
    profiles of elastic parameters in time (or depth)
    domain arranged in an input vector :math:`\mathbf{m}`
    of size :math:`n_{t_0} \times N`:

    .. math::
        d(t, \theta) =  \sum_{i=1}^N G_i(t, \theta) m_i(t) * w(t)

    where :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{G} \mathbf{w}

    On the other hand, pre-stack wavelet estimation aims at
    recovering the wavelet given knowledge of the band-limited
    seismic pre-stack data and the elastic parameter profiles.

    """
    ncp = get_array_module(theta)

    # define dtype to be used
    dtype = m.dtype  # ensure m.dtype rules that of operator
    theta = theta.astype(dtype)

    # Create vsvp profile
    vsvp = (
        vsvp
        if isinstance(vsvp, ncp.ndarray)
        else vsvp * ncp.ones(m.shape[0], dtype=dtype)
    )
    wavc = nwav // 2 if wavc is None else wavc
    nt0 = len(vsvp)
    ntheta = len(theta)

    # Create AVO operator
    if linearization == "akirich":
        G = akirichards(theta, vsvp, n=nt0)
    elif linearization == "fatti":
        G = fatti(theta, vsvp, n=nt0)
    elif linearization == "ps":
        G = ps(theta, vsvp, n=nt0)
    elif callable(linearization):
        G = linearization(theta, vsvp, n=nt0)
    else:
        logging.error("%s not an available linearization...", linearization)
        raise NotImplementedError(
            "%s not an available linearization..." % linearization
        )
    nG = len(G)
    G = [
        ncp.hstack([ncp.diag(G_[itheta] * ncp.ones(nt0, dtype=dtype)) for G_ in G])
        for itheta in range(ntheta)
    ]
    G = ncp.vstack(G).reshape(ntheta * nt0, nG * nt0)

    # Create derivative operator
    D = ncp.diag(0.5 * np.ones(nt0 - 1, dtype=dtype), k=1) - ncp.diag(
        0.5 * np.ones(nt0 - 1, dtype=dtype), k=-1
    )
    D[0] = D[-1] = 0
    D = get_block_diag(theta)(*([D] * nG))

    # Create infinite-reflectivity data
    M = ncp.dot(G, ncp.dot(D, m.T.ravel())).reshape(ntheta, nt0)
    Mconv = VStack(
        [
            MatrixMult(convmtx(M[itheta], nwav)[wavc : -nwav + wavc + 1], dtype=dtype)
            for itheta in range(ntheta)
        ]
    )
    return Mconv


def PrestackInversion(
    data,
    theta,
    wav,
    m0=None,
    linearization="akirich",
    explicit=False,
    simultaneous=False,
    epsI=None,
    epsR=None,
    dottest=False,
    returnres=False,
    epsRL1=None,
    kind="centered",
    vsvp=0.5,
    **kwargs_solver
):
    r"""Pre-stack linearized seismic inversion.

    Invert pre-stack seismic operator to retrieve a set of elastic property
    profiles from band-limited seismic pre-stack data (i.e., angle gathers).
    Depending on the choice of input parameters, inversion can be
    trace-by-trace with explicit operator or global with either
    explicit or linear operator.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Band-limited seismic post-stack data of size
        :math:`[(n_\text{lins} \times) \, n_{t_0} \times n_{\theta} \, (\times n_x \times n_y)]`
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of elements
        and centered to zero)
    m0 : :obj:`np.ndarray`, optional
        Background model of size :math:`[n_{t_0} \times n_{m}
        \,(\times n_x \times n_y)]`
    linearization : `{"akirich", "fatti", "PS"}` or :obj:`list`, optional
        * "akirich": Aki-Richards. See :py:func:`pylops.avo.avo.akirichards`.

        * "fatti": Fatti. See :py:func:`pylops.avo.avo.fatti`.

        * "PS": PS. See :py:func:`pylops.avo.avo.ps`.

        * List which is a combination of previous options (required only when ``m0 is None``).

    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preferred for small data)
    simultaneous : :obj:`bool`, optional
        Simultaneously invert entire data (``True``) or invert
        trace-by-trace (``False``) when using ``explicit`` operator
        (note that the entire data is always inverted when working
        with linear operator)
    epsI : :obj:`float` or :obj:`list`, optional
        Damping factor(s) for Tikhonov regularization term. If a list of
        :math:`n_{m}` elements is provided, the regularization term will have
        different strenght for each elastic property
    epsR : :obj:`float`, optional
        Damping factor for additional Laplacian regularization term
    dottest : :obj:`bool`, optional
        Apply dot-test
    returnres : :obj:`bool`, optional
        Return residuals
    epsRL1 : :obj:`float`, optional
        Damping factor for additional blockiness regularization term
    kind : :obj:`str`, optional
        Derivative kind (``forward`` or ``centered``).
    vsvp : :obj:`float` or :obj:`np.ndarray`
        :math:`V_S/V_P` ratio (constant or time/depth variant)
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.linalg.lstsq`
        solver (if ``explicit=True`` and  ``epsR=None``)
        or :py:func:`scipy.sparse.linalg.lsqr` solver (if ``explicit=False``
        and/or ``epsR`` is not ``None``))

    Returns
    -------
    minv : :obj:`np.ndarray`
        Inverted model of size :math:`[n_{t_0} \times n_{m}
        \,(\times n_x \times n_y)]`
    datar : :obj:`np.ndarray`
        Residual data (i.e., data - background data) of
        size :math:`[n_{t_0} \times n_{\theta} \,(\times n_x \times n_y)]`

    Notes
    -----
    The different choices of cost functions and solvers used in the
    seismic pre-stack inversion module follow the same convention of the
    seismic post-stack inversion module.

    Refer to :py:func:`pylops.avo.poststack.PoststackInversion` for
    more details.
    """
    ncp = get_array_module(data)

    # find out dimensions
    if m0 is None and linearization is None:
        raise NotImplementedError("either m0 or linearization " "must be provided")
    elif m0 is None:
        if isinstance(linearization, str):
            nm = _linearizations[linearization]
        else:
            nm = _linearizations[linearization[0]]
    else:
        nm = m0.shape[1]

    data_shape = data.shape
    data_ndim = data.ndim
    n_lins = 1
    multi = 0
    if not isinstance(linearization, str):
        n_lins = data_shape[0]
        data_shape = data_shape[1:]
        data_ndim -= 1
        multi = 1

    if data_ndim == 2:
        dims = 1
        nt0, ntheta = data_shape
        nspat = None
        nspatprod = nx = 1
    elif data_ndim == 3:
        dims = 2
        nt0, ntheta, nx = data_shape
        nspat = (nx,)
        nspatprod = nx
    else:
        dims = 3
        nt0, ntheta, nx, ny = data_shape
        nspat = (nx, ny)
        nspatprod = nx * ny
        data = data.reshape(nt0, ntheta, nspatprod)

    # check if background model and data have same shape
    if m0 is not None:
        if (
            nt0 != m0.shape[0]
            or (dims >= 2 and nx != m0.shape[2])
            or (dims == 3 and ny != m0.shape[3])
        ):
            raise ValueError("data and m0 must have same time and space axes")

    # create operator
    if isinstance(linearization, str):
        # single operator
        PPop = PrestackLinearModelling(
            wav,
            theta,
            vsvp=vsvp,
            nt0=nt0,
            spatdims=nspat,
            linearization=linearization,
            explicit=explicit,
            kind=kind,
        )
    else:
        # multiple operators
        if not isinstance(wav, (list, tuple)):
            wav = [
                wav,
            ] * n_lins
        PPop = [
            PrestackLinearModelling(
                w,
                theta,
                vsvp=vsvp,
                nt0=nt0,
                spatdims=nspat,
                linearization=lin,
                explicit=explicit,
            )
            for w, lin in zip(wav, linearization)
        ]
        if explicit:
            PPop = MatrixMult(
                np.vstack([Op.A for Op in PPop]), dims=nspat, dtype=PPop[0].A.dtype
            )
        else:
            PPop = VStack(PPop)

    if dottest:
        Dottest(
            PPop,
            n_lins * nt0 * ntheta * nspatprod,
            nt0 * nm * nspatprod,
            raiseerror=True,
            verb=True,
            backend=get_module_name(ncp),
        )

    # swap axes for explicit operator
    if explicit:
        data = data.swapaxes(0 + multi, 1 + multi)
        if m0 is not None:
            m0 = m0.swapaxes(0, 1)

    # invert model
    if epsR is None:
        # create and remove background data from original data
        datar = data.ravel() if m0 is None else data.ravel() - PPop * m0.ravel()
        # inversion without spatial regularization
        if explicit:
            if epsI is None and not simultaneous:
                # solve unregularized equations indipendently trace-by-trace
                minv = get_lstsq(data)(
                    PPop.A,
                    datar.reshape(n_lins * nt0 * ntheta, nspatprod).squeeze(),
                    **kwargs_solver
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
                PP = ncp.dot(PPop.A.T, PPop.A) + epsI * ncp.eye(
                    nt0 * nm, dtype=PPop.A.dtype
                )
                datarn = np.dot(PPop.A.T, datar.reshape(nt0 * ntheta, nspatprod))
                if not simultaneous:
                    # solve regularized normal eqs. trace-by-trace
                    minv = get_lstsq(data)(PP, datarn, **kwargs_solver)[0]
                else:
                    # solve regularized normal equations simultaneously
                    PPop_reg = MatrixMult(PP, dims=nspatprod)
                    if ncp == np:
                        minv = lsqr(PPop_reg, datarn.ravel(), **kwargs_solver)[0]
                    else:
                        minv = cgls(
                            PPop_reg,
                            datarn.ravel(),
                            x0=ncp.zeros(int(PPop_reg.shape[1]), PPop_reg.dtype),
                            **kwargs_solver
                        )[0]
            # else:
            #    # create regularized normal eqs. and solve them simultaneously
            #    PP = np.dot(PPop.A.T, PPop.A) + epsI * np.eye(nt0*nm)
            #    datarn = PPop.A.T * datar.reshape(nt0*ntheta, nspatprod)
            #    PPop_reg = MatrixMult(PP, dims=ntheta*nspatprod)
            #    minv = lstsq(PPop_reg, datarn.ravel(), **kwargs_solver)[0]
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
        # Create Thicknov regularization
        if epsI is not None:
            if isinstance(epsI, (list, tuple)):
                if len(epsI) != nm:
                    raise ValueError("epsI must be a scalar or a list of" "size nm")
                RegI = Diagonal(np.array(epsI), dims=(nt0, nm, nspatprod), dir=1)
            else:
                RegI = epsI * Identity(nt0 * nm * nspatprod)

        if epsRL1 is None:
            # L2 inversion with spatial regularization
            if dims == 1:
                Regop = SecondDerivative(nt0 * nm, dtype=PPop.dtype, dims=(nt0, nm))
            elif dims == 2:
                Regop = Laplacian((nt0, nm, nx), dirs=(0, 2), dtype=PPop.dtype)
            else:
                Regop = Laplacian((nt0, nm, nx, ny), dirs=(2, 3), dtype=PPop.dtype)
            if epsI is None:
                Regop = (Regop,)
                epsR = (epsR,)
            else:
                Regop = (Regop, RegI)
                epsR = (epsR, 1)
            minv = RegularizedInversion(
                PPop,
                Regop,
                data.ravel(),
                x0=m0.ravel() if m0 is not None else None,
                epsRs=epsR,
                returninfo=False,
                **kwargs_solver
            )
        else:
            # Blockiness-promoting inversion with spatial regularization
            if dims == 1:
                RegL1op = FirstDerivative(nt0 * nm, dtype=PPop.dtype)
                RegL2op = None
            elif dims == 2:
                RegL1op = FirstDerivative(
                    nt0 * nx * nm, dims=(nt0, nm, nx), dir=0, dtype=PPop.dtype
                )
                RegL2op = SecondDerivative(
                    nt0 * nx * nm, dims=(nt0, nm, nx), dir=2, dtype=PPop.dtype
                )
            else:
                RegL1op = FirstDerivative(
                    nt0 * nx * ny * nm, dims=(nt0, nm, nx, ny), dir=0, dtype=PPop.dtype
                )
                RegL2op = Laplacian((nt0, nm, nx, ny), dirs=(2, 3), dtype=PPop.dtype)
            if dims == 1:
                if epsI is not None:
                    RegL2op = (RegI,)
                    epsR = (1,)
            else:
                if epsI is None:
                    RegL2op = (RegL2op,)
                    epsR = (epsR,)
                else:
                    RegL2op = (RegL2op, RegI)
                    epsR = (epsR, 1)
            epsRL1 = (epsRL1,)
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
            minv = SplitBregman(
                PPop,
                (RegL1op,),
                data.ravel(),
                RegsL2=RegL2op,
                epsRL1s=epsRL1,
                epsRL2s=epsR,
                mu=mu,
                niter_outer=niter_outer,
                niter_inner=niter_inner,
                x0=None if m0 is None else m0.ravel(),
                **kwargs_solver
            )[0]

    # compute residual
    if returnres:
        if epsR is None:
            datar -= PPop * minv.ravel()
        else:
            datar = data.ravel() - PPop * minv.ravel()

    # re-swap axes for explicit operator
    if explicit:
        if m0 is not None:
            m0 = m0.swapaxes(0, 1)

    # reshape inverted model and residual data
    if dims == 1:
        if explicit:
            minv = minv.reshape(nm, nt0).swapaxes(0, 1)
            if returnres:
                datar = (
                    datar.reshape(n_lins, ntheta, nt0)
                    .squeeze()
                    .swapaxes(0 + multi, 1 + multi)
                )
        else:
            minv = minv.reshape(nt0, nm)
            if returnres:
                datar = datar.reshape(n_lins, nt0, ntheta).squeeze()
    elif dims == 2:
        if explicit:
            minv = minv.reshape(nm, nt0, nx).swapaxes(0, 1)
            if returnres:
                datar = (
                    datar.reshape(n_lins, ntheta, nt0, nx)
                    .squeeze()
                    .swapaxes(0 + multi, 1 + multi)
                )
        else:
            minv = minv.reshape(nt0, nm, nx)
            if returnres:
                datar = datar.reshape(n_lins, nt0, ntheta, nx).squeeze()
    else:
        if explicit:
            minv = minv.reshape(nm, nt0, nx, ny).swapaxes(0, 1)
            if returnres:
                datar = (
                    datar.reshape(n_lins, ntheta, nt0, nx, ny)
                    .squeeze()
                    .swapaxes(0 + multi, 1 + multi)
                )
        else:
            minv = minv.reshape(nt0, nm, nx, ny)
            if returnres:
                datar = datar.reshape(n_lins, nt0, ntheta, nx, ny).squeeze()

    if m0 is not None and epsR is None:
        minv = minv + m0

    if returnres:
        return minv, datar
    else:
        return minv
