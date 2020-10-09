import logging
import warnings
import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.signal import filtfilt
from scipy.ndimage.filters import convolve1d as sp_convolve1d

from pylops import Diagonal, Identity, Transpose
from pylops.signalprocessing import FFT, Fredholm1
from pylops.utils import dottest as Dottest
from pylops.optimization.leastsquares import PreconditionedInversion

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _MDC(G, nt, nv, dt=1., dr=1., twosided=True, fast=None, dtype=None,
         transpose=True, saveGt=True, conj=False, prescaled=False,
         _Identity=Identity, _Transpose=Transpose, _FFT=FFT,
         _Fredholm1=Fredholm1, args_Identity={}, args_Transpose={},
         args_FFT={}, args_Identity1={}, args_Transpose1={},
         args_FFT1={}, args_Fredholm1={}):
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries to
    MDC. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    warnings.warn('A new implementation of MDC is provided in v1.5.0. This '
                  'currently affects only the inner working of the operator, '
                  'end-users can continue using the operator in the same way. '
                  'Nevertheless, it is now recommended to start using the '
                  'operator with transpose=True, as this behaviour will '
                  'become default in version v2.0.0 and the behaviour with '
                  'transpose=False will be deprecated.', FutureWarning)

    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # transpose G
    if transpose:
        G = np.transpose(G, axes=(2, 0, 1))

    # find out dtype of G
    dtype = G[0, 0, 0].dtype
    rdtype = np.real(np.ones(1, dtype=dtype)).dtype

    # create Fredholm operator
    if prescaled:
        Frop = _Fredholm1(G, nv, saveGt=saveGt,
                          dtype=dtype, **args_Fredholm1)
    else:
        Frop = _Fredholm1(dr * dt * np.sqrt(nt) * G, nv, saveGt=saveGt,
                          dtype=dtype, **args_Fredholm1)
    if conj:
        Frop = Frop.conj()

    # create FFT operators
    nfmax, ns, nr = G.shape
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = _FFT(dims=(nt, nr, nv), dir=0, real=True,
               fftshift=twosided, dtype=rdtype, **args_FFT)
    F1op = _FFT(dims=(nt, ns, nv), dir=0, real=True,
                fftshift=False, dtype=rdtype, **args_FFT1)

    # create Identity operator to extract only relevant frequencies
    Iop = _Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                    inplace=True, dtype=dtype, **args_Identity)
    I1op = _Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                     inplace=True, dtype=dtype, **args_Identity1)
    F1opH = F1op.H
    I1opH = I1op.H

    # create transpose operator
    if transpose:
        dims = [nr, nt] if nv == 1 else [nr, nv, nt]
        axes = (1, 0) if nv == 1 else (2, 0, 1)
        Top = _Transpose(dims, axes, dtype=dtype, **args_Transpose)

        dims = [nt, ns] if nv == 1 else [nt, ns, nv]
        axes = (1, 0) if nv == 1 else (1, 2, 0)
        TopH = _Transpose(dims, axes, dtype=dtype, **args_Transpose1)

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    if transpose:
        MDCop = TopH * MDCop * Top

    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop


def MDC(G, nt, nv, dt=1., dr=1., twosided=True, fast=None,
        dtype=None, fftengine='numpy', transpose=True,
        saveGt=True, conj=False, usematmul=False, prescaled=False):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two datasets. If
    ``transpose=True``, model and data should be provided after flattening
    2- or 3-dimensional arrays of size :math:`[n_r (\times n_{vs}) \times n_t]`
    and :math:`[n_s (\times n_{vs}) \times n_t]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively. If ``transpose=False``, model and data
    should be provided after flattening 2- or 3-dimensional arrays of size
    :math:`[n_t \times n_r (\times n_{vs})]` and
    :math:`[n_t \times n_s (\times n_{vs})]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively.

    .. warning:: A new implementation of MDC is provided in v1.5.0. This
      currently affects only the inner working of the operator and end-users
      can use the operator in the same way as they used to do with the previous
      one. Nevertheless, it is now reccomended to use the operator with
      ``transpose=False``, as this behaviour will become default in version
      v2.0.0 and the behaviour with ``transpose=True`` will be deprecated.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]` if ``transpose=True``
        or size :math:`[n_{fmax} \times n_s \times n_r]` if ``transpose=False``
    nt : :obj:`int`
        Number of samples along time axis
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fast : :obj:`bool`, optional
        *Deprecated*, will be removed in v2.0.0
    dtype : :obj:`str`, optional
        *Deprecated*, will be removed in v2.0.0
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    transpose : :obj:`bool`, optional
        Transpose ``G`` and inputs such that time/frequency is placed in first
        dimension. This allows back-compatibility with v1.4.0 and older but
        will be removed in v2.0.0 where time/frequency axis will be required
        to be in first dimension for efficiency reasons.
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
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
        \mathscr{F}(x(t, r, v)) dr \Big)

    which is discretized as follows:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \sum_{i_r=0}^{n_r}
        (\sqrt{n_t} * d_t * d_r) G(f, s, i_r) \mathscr{F}(x(t, i_r, v)) \Big)

    where :math:`(\sqrt{n_t} * d_t * d_r)` is not applied if ``prescaled=True``.

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
    return _MDC(G, nt, nv, dt=dt, dr=dr, twosided=twosided, fast=fast,
                dtype=dtype, transpose=transpose, saveGt=saveGt,
                conj=conj, prescaled=prescaled,
                args_FFT={'engine': fftengine},
                args_Fredholm1={'usematmul': usematmul})


def MDD(G, d, dt=0.004, dr=1., nfmax=None, wav=None,
        twosided=True, causality_precond=False, adjoint=False,
        psf=False, dtype='float64', dottest=False,
        saveGt=True, add_negative=True, smooth_precond=0, **kwargs_lsqr):
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
        :math:`[n_s \times n_r \times 2*n_t-1]` for ``twosided=True`` and
        ``add_negative=False``
        (with both positive and negative times)
    d : :obj:`numpy.ndarray`
        Data in time domain :math:`[n_s (\times n_vs) \times n_t]` if
        ``twosided=False`` or ``twosided=True`` and ``add_negative=True``
        (with only positive times) or size
        :math:`[n_s (\times n_vs) \times 2*n_t-1]` if ``twosided=True``
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
    adjoint : :obj:`bool`, optional
        Compute and return adjoint(s)
    psf : :obj:`bool`, optional
        Compute and return Point Spread Function (PSF) and its inverse
    dtype : :obj:`bool`, optional
        Type of elements in input array.
    dottest : :obj:`bool`, optional
        Apply dot-test
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    **kwargs_lsqr
        Arbitrary keyword arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver

    Returns
    -------
    minv : :obj:`numpy.ndarray`
        Inverted model of size :math:`[n_r (\times n_{vs}) \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r (\times n_vs) \times 2*n_t-1]` for ``twosided=True``
    madj : :obj:`numpy.ndarray`
        Adjoint model of size :math:`[n_r (\times n_{vs}) \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r (\times n_r) \times 2*n_t-1]` for ``twosided=True``
    psfinv : :obj:`numpy.ndarray`
        Inverted psf of size :math:`[n_r \times n_r \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \times n_r \times 2*n_t-1]` for ``twosided=True``
    psfadj : :obj:`numpy.ndarray`
        Adjoint psf of size :math:`[n_r \times n_r \times n_t]`
        for ``twosided=False`` or
        :math:`[n_r \times n_r \times 2*n_t-1]` for ``twosided=True``

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
        nfmax_allowed = int(np.ceil((nt2+1)/2))
    else:
        nt2 = nt
        nfmax_allowed = nt

    # Fix nfmax to be at maximum equal to half of the size of fft samples
    if nfmax is None or nfmax > nfmax_allowed:
        nfmax = nfmax_allowed
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    # Add negative part to data and model
    if twosided and add_negative:
        G = np.concatenate((np.zeros((ns, nr, nt - 1)), G), axis=-1)
        d = np.concatenate((np.squeeze(np.zeros((ns, nv, nt - 1))), d),
                           axis=-1)

    # Bring kernel to frequency domain
    Gfft = np.fft.rfft(G, nt2, axis=-1)
    Gfft = Gfft[..., :nfmax]

    # Bring frequency/time to first dimension
    Gfft = np.moveaxis(Gfft, -1, 0)
    d = np.moveaxis(d, -1, 0)
    if psf:
        G = np.moveaxis(G, -1, 0)

    # Define MDC linear operator
    MDCop = MDC(Gfft, nt2, nv=nv, dt=dt, dr=dr, twosided=twosided,
                transpose=False, saveGt=saveGt)
    if psf:
        PSFop = MDC(Gfft, nt2, nv=nr, dt=dt, dr=dr, twosided=twosided,
                    transpose=False, saveGt=saveGt)
    if dottest:
        Dottest(MDCop, nt2*ns*nv, nt2*nr*nv, verb=True)
        if psf:
            Dottest(PSFop, nt2 * ns * nr, nt2 * nr * nr, verb=True)

    # Adjoint
    if adjoint:
        madj = MDCop.H * d.flatten()
        madj = np.squeeze(madj.reshape(nt2, nr, nv))
        madj = np.moveaxis(madj, 0, -1)
        if psf:
            psfadj = PSFop.H * G.flatten()
            psfadj = np.squeeze(psfadj.reshape(nt2, nr, nr))
            psfadj = np.moveaxis(psfadj, 0, -1)

    # Inverse
    if twosided and causality_precond:
        P = np.ones((nt2, nr, nv))
        P[:nt - 1] = 0
        if smooth_precond > 0:
            P = filtfilt(np.ones(smooth_precond)/smooth_precond, 1, P, axis=0)
        Pop = Diagonal(P)
        minv = PreconditionedInversion(MDCop, Pop, d.flatten(),
                                       returninfo=False, **kwargs_lsqr)
    else:
        minv = lsqr(MDCop, d.flatten(), **kwargs_lsqr)[0]
    minv = np.squeeze(minv.reshape(nt2, nr, nv))
    minv = np.moveaxis(minv, 0, -1)
    if wav is not None:
        minv = sp_convolve1d(minv, wav, axis=-1)

    if psf:
        psfinv = lsqr(PSFop, G.flatten(), **kwargs_lsqr)[0]
        psfinv = np.squeeze(psfinv.reshape(nt2, nr, nr))
        psfinv = np.moveaxis(psfinv, 0, -1)
        if wav is not None:
            psfinv = sp_convolve1d(psfinv, wav, axis=-1)

    if adjoint and psf:
        return minv, madj, psfinv, psfadj
    elif adjoint:
        return minv, madj
    elif psf:
        return minv, psfinv
    else:
        return minv
