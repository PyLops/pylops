__all__ = [
    "MDC",
    "MDD",
]

import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr

from pylops import Diagonal, Identity, LinearOperator, Transpose
from pylops.optimization.basic import cgls
from pylops.optimization.leastsquares import preconditioned_inversion
from pylops.signalprocessing import FFT, Fredholm1
from pylops.utils import dottest as Dottest
from pylops.utils.backend import (
    get_array_module,
    get_fftconvolve,
    get_module_name,
    to_cupy_conditional,
)
from pylops.utils.typing import NDArray


def _MDC(
    G: NDArray,
    nt: int,
    nv: int,
    dt: float = 1.0,
    dr: float = 1.0,
    twosided: bool = True,
    saveGt: bool = True,
    conj: bool = False,
    prescaled: bool = False,
    _Identity=Identity,
    _Transpose=Transpose,
    _FFT=FFT,
    _Fredholm1=Fredholm1,
    args_Identity: Optional[dict] = None,
    args_FFT: Optional[dict] = None,
    args_Identity1: Optional[dict] = None,
    args_FFT1: Optional[dict] = None,
    args_Fredholm1: Optional[dict] = None,
) -> LinearOperator:
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries (e.g., pylops-distributed) to
    MDC. It operates in the same way as public method
    (MDC) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    if args_Identity is None:
        args_Identity = {}
    if args_FFT is None:
        args_FFT = {}
    if args_Identity1 is None:
        args_Identity1 = {}
    if args_FFT1 is None:
        args_FFT1 = {}
    if args_Fredholm1 is None:
        args_Fredholm1 = {}

    if twosided and nt % 2 == 0:
        raise ValueError("nt must be odd number")

    # find out dtype of G
    dtype = G[0, 0, 0].dtype
    rdtype = np.real(np.ones(1, dtype=dtype)).dtype

    # create Fredholm operator
    if prescaled:
        Frop = _Fredholm1(G, nv, saveGt=saveGt, dtype=dtype, **args_Fredholm1)
    else:
        Frop = _Fredholm1(
            dr * dt * np.sqrt(nt) * G, nv, saveGt=saveGt, dtype=dtype, **args_Fredholm1
        )
    if conj:
        Frop = Frop.conj()

    # create FFT operators
    nfmax, ns, nr = G.shape
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning("nfmax set equal to ceil[(nt+1)/2=%d]", nfmax)

    Fop = _FFT(
        dims=(nt, nr, nv),
        axis=0,
        real=True,
        ifftshift_before=twosided,
        dtype=rdtype,
        **args_FFT
    )
    F1op = _FFT(
        dims=(nt, ns, nv),
        axis=0,
        real=True,
        ifftshift_before=False,
        dtype=rdtype,
        **args_FFT1
    )

    # create Identity operator to extract only relevant frequencies
    Iop = _Identity(
        N=nfmax * nr * nv, M=nfft * nr * nv, inplace=True, dtype=dtype, **args_Identity
    )
    I1op = _Identity(
        N=nfmax * ns * nv, M=nfft * ns * nv, inplace=True, dtype=dtype, **args_Identity1
    )
    F1opH = F1op.H
    I1opH = I1op.H

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop

    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop


def MDC(
    G: NDArray,
    nt: int,
    nv: int,
    dt: float = 1.0,
    dr: float = 1.0,
    twosided: bool = True,
    fftengine: str = "numpy",
    saveGt: bool = True,
    conj: bool = False,
    usematmul: bool = False,
    prescaled: bool = False,
    name: str = "M",
) -> LinearOperator:
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two datasets.
    Model and data should be provided after flattening 2- or 3-dimensional arrays
    of size :math:`[n_t \times n_r \;(\times n_{vs})]` and
    :math:`[n_t \times n_s \;(\times n_{vs})]` (or :math:`2n_t-1` for
    ``twosided=True``), respectively.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_{f_\text{max}} \times n_s \times n_r]`
    nt : :obj:`int`
        Number of samples along time axis for model and data (note that this
        must be equal to :math:`2n_t-1` when working with ``twosided=True``.
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis :math:`\Delta t`
    dr : :obj:`float`, optional
        Sampling of receiver integration axis :math:`\Delta r`
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``scipy`` or ``fftw``)
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G.H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G.H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
        Refer to Fredholm1 documentation for details.
    prescaled : :obj:`bool`, optional
        Apply scaling to kernel (``False``) or not (``False``) when performing
        spatial and temporal summations. In case ``prescaled=True``, the
        kernel is assumed to have been pre-scaled when passed to the MDC
        routine.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Raises
    ------
    ValueError
        If ``nt`` is even and ``twosided=True``

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained
    operator [1]_. It is composed of a forward Fourier transform,
    a multi-dimensional integration, and an inverse Fourier transform:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \int_S G(f, s, r)
        \mathscr{F}(x(t, r, v))\,\mathrm{d}r \Big)

    which is discretized as follows:

    .. math::
        y(t, s, v) = \sqrt{n_t} \Delta t \Delta r\mathscr{F}^{-1} \Big( \sum_{i_r=0}^{n_r}
        G(f, s, i_r) \mathscr{F}(x(t, i_r, v)) \Big)

    where :math:`\sqrt{n_t} \Delta t \Delta r` is not applied if ``prescaled=True``.

    This operation can be discretized and performed by means of a
    linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{G} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along
    the time axis and :math:`\mathbf{G}` is the multi-dimensional
    convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophysical Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    MOp = _MDC(
        G,
        nt,
        nv,
        dt=dt,
        dr=dr,
        twosided=twosided,
        saveGt=saveGt,
        conj=conj,
        prescaled=prescaled,
        args_FFT={"engine": fftengine},
        args_FFT1={"engine": fftengine},
        args_Fredholm1={"usematmul": usematmul},
    )
    MOp.name = name
    return MOp


def MDD(
    G: NDArray,
    d: NDArray,
    dt: float = 0.004,
    dr: float = 1.0,
    nfmax: Optional[int] = None,
    wav: Optional[NDArray] = None,
    twosided: bool = True,
    causality_precond: bool = False,
    adjoint: bool = False,
    psf: bool = False,
    dottest: bool = False,
    saveGt: bool = True,
    add_negative: bool = True,
    smooth_precond: int = 0,
    fftengine: str = "numpy",
    **kwargs_solver
) -> Union[
    Tuple[NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray, NDArray],
]:
    r"""Multi-dimensional deconvolution.

    Solve multi-dimensional deconvolution problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in time domain of size
        :math:`[n_s \times n_r \times n_t]` for ``twosided=False`` or
        ``twosided=True`` and ``add_negative=True``
        (with only positive times) or size
        :math:`[n_s \times n_r \times 2n_t-1]` for ``twosided=True`` and
        ``add_negative=False``
        (with both positive and negative times)
    d : :obj:`numpy.ndarray`
        Data in time domain :math:`[n_s \,(\times n_{vs}) \times n_t]` if
        ``twosided=False`` or ``twosided=True`` and ``add_negative=True``
        (with only positive times) or size
        :math:`[n_s \,(\times n_{vs}) \times 2n_t-1]` if ``twosided=True``
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    wav : :obj:`numpy.ndarray`, optional
        Wavelet to convolve to the inverted model and psf
        (must be centered around its index in the middle of the array).
        If ``None``, the outputs of the inversion are returned directly.
    twosided : :obj:`bool`, optional
        MDC operator and data both negative and positive time (``True``)
        or only positive (``False``)
    add_negative : :obj:`bool`, optional
        Add negative side to MDC operator and data (``True``) or not
        (``False``)- operator and data are already provided with both positive
        and negative sides. To be used only with ``twosided=True``.
    causality_precond : :obj:`bool`, optional
        Apply causality mask (``True``) or not (``False``)
    smooth_precond : :obj:`int`, optional
        Lenght of smoothing to apply to causality preconditioner
    adjoint : :obj:`bool`, optional
        Compute and return adjoint(s)
    psf : :obj:`bool`, optional
        Compute and return Point Spread Function (PSF) and its inverse
    dottest : :obj:`bool`, optional
        Apply dot-test
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G.H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G.H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``scipy`` or ``fftw``)
    **kwargs_solver
        Arbitrary keyword arguments for chosen solver
        (:py:func:`scipy.sparse.linalg.cg` and
        :py:func:`pylops.optimization.solver.cg` are used as default for numpy
        and cupy `data`, respectively)

    Returns
    -------
    minv : :obj:`numpy.ndarray`
        Inverted model of size :math:`[n_r \,(\times n_{vs}) \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \,(\times n_vs) \times 2n_t-1]` for ``twosided=True``
    madj : :obj:`numpy.ndarray`
        Adjoint model of size :math:`[n_r \,(\times n_{vs}) \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \,(\times n_r) \times 2n_t-1]` for ``twosided=True``
    psfinv : :obj:`numpy.ndarray`
        Inverted psf of size :math:`[n_r \times n_r \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \times n_r \times 2n_t-1]` for ``twosided=True``
    psfadj : :obj:`numpy.ndarray`
        Adjoint psf of size :math:`[n_r \times n_r \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \times n_r \times 2n_t-1]` for ``twosided=True``

    See Also
    --------
    MDC : Multi-dimensional convolution

    Notes
    -----
    Multi-dimensional deconvolution (MDD) is a mathematical ill-solved problem,
    well-known in the image processing and geophysical community [1]_.

    MDD aims at removing the effects of a Multi-dimensional Convolution
    (MDC) kernel or the so-called blurring operator or point-spread
    function (PSF) from a given data. It can be written as

    .. math::
        \mathbf{d}= \mathbf{D} \mathbf{m}

    or, equivalently, by means of its normal equation

    .. math::
        \mathbf{m}= (\mathbf{D}^H\mathbf{D})^{-1} \mathbf{D}^H\mathbf{d}

    where :math:`\mathbf{D}^H\mathbf{D}` is the PSF.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophyscial Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    ncp = get_array_module(d)

    ns, nr, nt = G.shape
    if len(d.shape) == 2:
        nv = 1
    else:
        nv = d.shape[1]
    if twosided:
        if add_negative:
            nt2 = 2 * nt - 1
        else:
            nt2 = nt
            nt = (nt2 + 1) // 2
        nfmax_allowed = int(np.ceil((nt2 + 1) / 2))
    else:
        nt2 = nt
        nfmax_allowed = nt

    # Fix nfmax to be at maximum equal to half of the size of fft samples
    if nfmax is None or nfmax > nfmax_allowed:
        nfmax = nfmax_allowed
        logging.warning("nfmax set equal to ceil[(nt+1)/2=%d]", nfmax)

    # Add negative part to data and model
    if twosided and add_negative:
        G = np.concatenate((ncp.zeros((ns, nr, nt - 1)), G), axis=-1)
        d = np.concatenate((np.squeeze(np.zeros((ns, nv, nt - 1))), d), axis=-1)
    # Bring kernel to frequency domain
    Gfft = np.fft.rfft(G, nt2, axis=-1)
    Gfft = Gfft[..., :nfmax]

    # Bring frequency/time to first dimension
    Gfft = np.moveaxis(Gfft, -1, 0)
    d = np.moveaxis(d, -1, 0)
    if psf:
        G = np.moveaxis(G, -1, 0)

    # Define MDC linear operator
    MDCop = MDC(
        Gfft,
        nt2,
        nv=nv,
        dt=dt,
        dr=dr,
        twosided=twosided,
        saveGt=saveGt,
        fftengine=fftengine,
    )
    if psf:
        PSFop = MDC(
            Gfft,
            nt2,
            nv=nr,
            dt=dt,
            dr=dr,
            twosided=twosided,
            saveGt=saveGt,
            fftengine=fftengine,
        )
    if dottest:
        Dottest(
            MDCop, nt2 * ns * nv, nt2 * nr * nv, verb=True, backend=get_module_name(ncp)
        )
        if psf:
            Dottest(
                PSFop,
                nt2 * ns * nr,
                nt2 * nr * nr,
                verb=True,
                backend=get_module_name(ncp),
            )

    # Adjoint
    if adjoint:
        madj = MDCop.H * d.ravel()
        madj = np.squeeze(madj.reshape(nt2, nr, nv))
        madj = np.moveaxis(madj, 0, -1)
        if psf:
            psfadj = PSFop.H * G.ravel()
            psfadj = np.squeeze(psfadj.reshape(nt2, nr, nr))
            psfadj = np.moveaxis(psfadj, 0, -1)

    # Inverse
    if twosided and causality_precond:
        P = np.ones((nt2, nr, nv))
        P[: nt - 1] = 0
        if smooth_precond > 0:
            P = filtfilt(np.ones(smooth_precond) / smooth_precond, 1, P, axis=0)
        P = to_cupy_conditional(d, P)
        Pop = Diagonal(P)
        minv = preconditioned_inversion(MDCop, d.ravel(), Pop, **kwargs_solver)[0]
    else:
        if ncp == np and "callback" not in kwargs_solver:
            minv = lsqr(MDCop, d.ravel(), **kwargs_solver)[0]
        else:
            minv = cgls(
                MDCop,
                d.ravel(),
                ncp.zeros(int(MDCop.shape[1]), dtype=MDCop.dtype),
                **kwargs_solver
            )[0]
    minv = np.squeeze(minv.reshape(nt2, nr, nv))
    minv = np.moveaxis(minv, 0, -1)

    if wav is not None:
        wav1 = wav.copy()
        for _ in range(minv.ndim - 1):
            wav1 = wav1[ncp.newaxis]
        minv = get_fftconvolve(d)(minv, wav1, mode="same")

    if psf:
        if ncp == np:
            psfinv = lsqr(PSFop, G.ravel(), **kwargs_solver)[0]
        else:
            psfinv = cgls(
                PSFop,
                G.ravel(),
                ncp.zeros(int(PSFop.shape[1]), dtype=PSFop.dtype),
                **kwargs_solver
            )[0]
        psfinv = np.squeeze(psfinv.reshape(nt2, nr, nr))
        psfinv = np.moveaxis(psfinv, 0, -1)
        if wav is not None:
            wav1 = wav.copy()
            for _ in range(psfinv.ndim - 1):
                wav1 = wav1[np.newaxis]
            psfinv = get_fftconvolve(d)(psfinv, wav1, mode="same")
    if adjoint and psf:
        return minv, madj, psfinv, psfadj
    elif adjoint:
        return minv, madj
    elif psf:
        return minv, psfinv
    else:
        return minv
