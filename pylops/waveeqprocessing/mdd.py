import logging
import warnings
import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.ndimage.filters import convolve1d as sp_convolve1d

from pylops import LinearOperator, Diagonal, Identity, Transpose
from pylops.signalprocessing import FFT, Fredholm1
from pylops.utils import dottest as Dottest
from pylops.optimization.leastsquares import PreconditionedInversion

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def MDC(G, nt, nv, dt=1., dr=1., twosided=True, fast=None,
        dtype=None, fftengine='numpy', transpose=True):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two datasets. If
    ``transpose=True``, model and data should be provided after flattening
    2- or 3-dimensional arrays of size :math:`[n_r \times n_{vs} \times n_t]`
    and :math:`[n_s \times n_{vs} \times n_t]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively. If ``transpose=False``, model and data
    should be provided after flattening 2- or 3-dimensional arrays of size
    :math:`[n_t \times n_r \times n_{vs}]` and
    :math:`[n_t \times n_s \times n_{vs}]` (or :math:`2*n_t-1` for
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
        :math:`[\times n_s \times n_r \times n_{fmax}]` if ``transpose=True``
        or size :math:`[n_{fmax} \times n_s \times n_r]` if ``transpose=False``
    nt : :obj:`int`
        Number of samples along time axis
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
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
    transpose : :obj:`str`, optional
        Transpose ``G`` and inputs such that time/frequency is placed in first
        dimension. This will be removed in v2.0.0 and time/frequency axis will
        be required to be in first dimension

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained
    operator [1]_. It is composed of a forward Fourier transform,
    a multi-dimensional integration, and an inverse Fourier transform:

    .. math::
        y(f, s, v) = \mathscr{F}^{-1} \Big( \int_S R(f, s, r)
        \mathscr{F}(x(f, r, v)) dr \Big)

    This operation can be discretized and performed by means of a
    linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{R} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along
    the time axis and :math:`\mathbf{R}` is the multi-dimensional
    convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophysical Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    warnings.warn('A new implementation of MDC is provided in v1.5.0. This'
                  'currently affects only the inner working of the operator '
                  'and end-users can use the operator in the same way as they '
                  'used to do with the previous one. Nevertheless, it is now '
                  'reccomended to use the operator with transpose=True, as '
                  'this behaviour will become default in version v2.0.0 and '
                  'the behaviour with transpose=False will be deprecated.',
                  FutureWarning)

    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # transpose G
    if transpose:
        G = np.transpose(G, axes=(2, 0, 1))

    # create Fredholm operator
    dtype = G[0, 0, 0].dtype
    fdtype = (G[0, 0, 0] + 1j*G[0, 0, 0]).dtype
    Frop = Fredholm1(dr*dt*np.sqrt(nt)*G, nv, usematmul=False, dtype=fdtype)

    # create FFT operators
    nfmax, ns, nr = G.shape
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt+1)/2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = FFT(dims=(nt, nr, nv), dir=0, real=True,
              fftshift=twosided, engine=fftengine, dtype=fdtype)
    F1op = FFT(dims=(nt, ns, nv), dir=0, real=True,
               fftshift=False, engine=fftengine, dtype=fdtype)

    # create Identity operator to extract only relevant frequencies
    Iop = Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                   inplace=True, dtype=dtype)
    I1op = Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                    inplace=True, dtype=dtype)
    F1opH = F1op.H
    I1opH = I1op.H

    # create transpose operator
    if transpose:
        dims = [nr, nt] if nv == 1 else [nr, nv, nt]
        axes = (1, 0) if nv == 1 else (2, 0, 1)
        Top = Transpose(dims, axes, dtype=dtype)

        dims = [nt, ns] if nv == 1 else [nt, ns, nv]
        axes = (1, 0) if nv == 1 else (1, 2, 0)
        TopH = Transpose(dims, axes, dtype=dtype)

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    if transpose:
        MDCop = TopH * MDCop * Top
    return MDCop


def MDD(G, d, dt=0.004, dr=1., nfmax=None, wav=None,
        twosided=True, add_negative=True,
        causality_precond=False, adjoint=False,
        psf=False, dtype='float64',
        dottest=False, **kwargs_lsqr):
    r"""Multi-dimensional deconvolution.

    Solve multi-dimensional deconvolution problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in time domain of size
        :math:`[n_s \times n_r \times n_t]` for ``twosided=False``
        (with only positive times) or size
        :math:`[n_s \times n_r \times 2*n_t-1]` for ``twosided=True``
        (with both positive and negative times)
    d : :obj:`numpy.ndarray`
        Data in time domain :math:`[n_s (\times n_vs) \times n_t]`
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    twosided : :obj:`bool`, optional
        MDC operator and data both negative and positive time (``True``)
        or only positive (``False``)
    add_negative : :obj:`bool`, optional
        Add negative side to MDC operator and data (``True``) or already
        provided with both positve and negative sides (``False``)
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
        ns, nt = d.shape
        nv = 1
    else:
        ns, nv, nt = d.shape
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
                transpose=False)
    if psf:
        PSFop = MDC(Gfft, nt2, nv=nr, dt=dt, dr=dr, twosided=twosided,
                    transpose=False)
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
