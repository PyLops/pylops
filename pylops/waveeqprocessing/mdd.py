import logging
import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.ndimage.filters import convolve1d as sp_convolve1d

from pylops import LinearOperator, Diagonal
from pylops.utils import dottest as Dottest
from pylops.optimization.leastsquares import PreconditionedInversion

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class MDC(LinearOperator):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between 3D seismic data.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel  in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]`
    nt : :obj:`int`
        Number of samples along time axis
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`
        Sampling of time integration axis
    dr : :obj:`float`
        Sampling of receiver integration axis
    twosided : :obj:`bool`
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fast : :obj:`bool`
        Fast application of MDC when model has only one virtual
        source (``True``) or not (``False``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of samples along receiver axis
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (True) or not (False)

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained operator [1]_.
    It is composed of a forward Fourier transform, a multi-dimensional
    integration, and an inverse Fourier transform:

    .. math::
        y(s,v,f) = \int_S R(s,r,f) x(r,v,f) dr

    This operation can be discretized and performed by means of a linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{R} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along the time axis
    and :math:`\mathbf{R}` is the multi-dimensional convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a systematic comparison",
       Geophyscial Journal International, vol. 185, pp. 1335-1364. 2011.

    """
    def __init__(self, G, nt, nv, dt=1., dr=1., twosided=True, fast=False, dtype='float32'):
        if twosided and nt % 2 == 0:
            raise ValueError('nt must be odd number')
        self.G = G
        self.ns, self.nr, self.nfmax = G.shape

        self.nt = nt
        self.nv = nv
        self.dt = dt
        self.dr = dr

        self.shape = (self.ns*self.nv*self.nt, self.nr*self.nv*self.nt)
        self.twosided = twosided
        self.fast = fast
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.squeeze(np.reshape(x, (self.nr, self.nv, self.nt)))
        if self.twosided:
            x = np.fft.ifftshift(x, axes=-1)
        x = np.sqrt(1./self.nt)*np.fft.rfft(x, self.nt, axis=-1)
        x = x[..., :self.nfmax]

        if self.nv == 1 and self.fast:
            y = self.dr * self.dt * np.sqrt(self.nt) * \
                np.sum(self.G * np.tile(x, [self.ns, 1, 1]), axis=1)
        else:
            y = np.squeeze(np.zeros((self.ns, self.nv, x.shape[-1]), dtype=np.complex128))
            for it in range(self.nfmax):
                y[..., it] = self.dr * self.dt * np.sqrt(self.nt) * \
                             np.dot(self.G[:, :, it], x[..., it])

        y = np.real(np.fft.irfft(y, self.nt, axis=-1)* np.sqrt(self.nt))
        y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        x = np.squeeze(np.reshape(x, (self.ns, self.nv, self.nt)))
        x = np.sqrt(1./self.nt)*np.fft.rfft(x, self.nt, axis=-1)
        x = x[..., :self.nfmax]

        if self.nv == 1 and self.fast:
            y = self.dr * self.dt * np.sqrt(self.nt) * \
                np.sum(np.conj(self.G) * np.tile(x[:, np.newaxis, :], [1, self.nr, 1]), axis=0)
        else:
            y = np.squeeze(np.zeros((self.nr, self.nv, x.shape[-1]), dtype=np.complex128))
            for it in range(self.nfmax):
                y[..., it] = self.dr * self.dt * np.sqrt(self.nt) * \
                            np.dot(np.conj(self.G[:, :, it].T), x[..., it])
        y = np.fft.irfft(y, self.nt, axis=-1)* np.sqrt(self.nt)
        if self.twosided:
            y = np.fft.fftshift(y, axes=-1)
        y = np.real(y)
        y = np.ndarray.flatten(y)
        return y


def MDD(G, d, dt=0.004, dr=1., nfmax=None, wav=None,
        twosided=True, causality_precond=False, adjoint=False,
        psf=False, dtype='complex64',
        dottest=False, **kwargs_lsqr):
    r"""Multi-dimensional deconvolution.

    Solve multi-dimensional deconvolution problem using :py:func:`scipy.sparse.linalg.lsqr`
    iterative solver.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel  in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]`
    d : :obj:`numpy.ndarray`
        Data in time domain :math:`[ns (\times nr) \times nt]`
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``)
        or only positive (``False``)
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
        Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` solver

    Returns
    ----------
    minv : :obj:`numpy.ndarray`
        Inverted model.
    madj : :obj:`numpy.ndarray`
        Adjoint model.
    psfinv : :obj:`numpy.ndarray`
        Inverted psf.
    psfadj : :obj:`numpy.ndarray`
        Adjoint psf.

    See Also
    --------
    MDC : Multi-dimensional convolution

    Notes
    -----
    Multi-dimensional deconvolution (MDD) is a mathematical ill-solved problem,
    well-known in the image processing and geophysical community [1]_.

    MDD aims at removing the effects of a Multi-dimensional Convolution (MDC) kernel
    or the so-called blurring operator or point-spread function (PSF) from a given data.
    It can be written as

    .. math::
        \mathbf{d}= \mathbf{D} \mathbf{m}

    or, equivalently, by means of its normal equation

    .. math::
        \mathbf{m}= (\mathbf{D}^H\mathbf{D})^{-1} \mathbf{D}^H\mathbf{d}

    where :math:`\mathbf{D}^H\mathbf{D}` is the PSF.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker, J.,
       Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry by crosscorrelation
       and by multi-dimensional deconvolution: a systematic comparison",
       Geophyscial Journal International, vol. 185, pp. 1335-1364. 2011.

    """
    ns, nr, nt = G.shape
    if len(d.shape) == 2:
        ns, nt = d.shape
        nv = 1
    else:
        ns, nv, nt = d.shape
    nt2 = nt if twosided == False else 2 * nt - 1

    # Fix nfmax to be at maximum equal to half of the size of fft samples
    if nfmax == None or nfmax > np.ceil((nt2 + 1) / 2):
        nfmax = int(np.ceil((nt2+1)/2))
        logging.warning('nfmax set equal to (nt+1)/2=%d' % nfmax)

    # Add negative part to data and model
    if twosided:
        G = np.concatenate((np.zeros((ns, nr, nt - 1)), G), axis=-1)
        d = np.concatenate((np.squeeze(np.zeros((ns, nv, nt - 1))), d), axis=-1)

    # Define MDC linear operator
    Gfft = np.fft.rfft(G, nt2, axis=-1)
    Gfft = Gfft[..., :nfmax]

    MDCop = MDC(Gfft, nt2, nv=nv, dt=dt, dr=dr, twosided=twosided, dtype=dtype)
    if psf:
        PSFop = MDC(Gfft, nt2, nv=nr, dt=dt, dr=dr, twosided=twosided, dtype=dtype)
    if dottest:
        Dottest(MDCop, nt2*ns*nv, nt2*nr*nv, verb=True)
        if psf:
            Dottest(PSFop, nt2 * ns * nr, nt2 * nr * nr, verb=True)

    # Adjoint
    if adjoint:
        madj = MDCop.H * d.flatten()
        madj = np.squeeze(madj.reshape(nr, nv, nt2))
        if psf:
            psfadj = PSFop.H * G.flatten()
            psfadj = np.squeeze(psfadj.reshape(nr, nr, nt2))

    # Inverse
    if twosided and causality_precond:
        P = np.ones((nr, nv, nt2))
        P[:, :, :nt - 1] = 0
        Pop = Diagonal(P)
        minv = PreconditionedInversion(MDCop, Pop, d.flatten(), returninfo=False, **kwargs_lsqr)
    else:
        minv = lsqr(MDCop, d.flatten(), **kwargs_lsqr)[0]
    minv = np.squeeze(minv.reshape(nr, nv, nt2))
    if wav is not None:
        minv = sp_convolve1d(minv, wav, axis=-1)

    if psf:
        psfinv = lsqr(PSFop, G.flatten(), **kwargs_lsqr)[0]
        psfinv = np.squeeze(psfinv.reshape(nr, nr, nt2))
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
