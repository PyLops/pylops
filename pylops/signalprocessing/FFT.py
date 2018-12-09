import numpy as np
from pylops import LinearOperator


class FFT(LinearOperator):
    r"""One dimensional Fast-Fourier Transform.

    Apply Fast-Fourier Transform (FFT) along a specific direction ``dir`` of a
    multi-dimensional array of size ``dim``. Note that the FFT operator is a simple
    overload to the numpy :py:func:`numpy.fft.fft` or (:py:func:`numpy.fft.rfft` for
    real models) in forward mode and to the numpy :py:func:`numpy.fft.ifft` or
    (:py:func:`numpy.fft.irfft` for real models) in adjoint mode, however
    scaling is taken into account differently to guarantee that the operator
    is passing the dot-test.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dir : :obj:`int`, optional
        Direction along which FFT is applied.
    nfft : :obj:`int`, optional
        Number of samples in Fourier Transform (same as input if ``nfft=None``)
    sampling : :obj:`float`, optional
        Sampling step ``dt``.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (True) or not (False).
        Used to enforce that the output of adjoint of a real model is real.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (True) or not (False)

    Raises
    ------
    ValueError
        If ``dims`` is not provided and if ``dir`` is bigger than ``len(dims)``

    Notes
    -----
    The FFT operator applies the forward Fourier transform to a signal :math:`d(t)` in forward mode:

    .. math::
        D(f) = \mathscr{F} (d) = \int d(t) e^{-j2\pi ft} dt

    Similarly, the inverse Fourier transform is applied to the Fourier spectrum
    :math:`D(f)` in adjoint mode:

    .. math::
        d(t) = \mathscr{F}^{-1} (D) = \int D(f) e^{j2\pi ft} df

    Both operators are effectively discretized and solved by a fast iterative algorithm
    known as Fast Fourier Transform.

    Note that the FFT operator is a special operator in that the adjoint adjoint is also the
    inverse of the forward mode. Moreover, in case of real signal in time domain, the Fourier
    transform in Hermitian.

    It is possible to store the values of the Fourier transformed signal at positive frequencies
    as those at negative frequencies are simply their complex conjugate. However as the operation
    of removing the negative part of the frequency axis in forward and adding complex conjugate
    version in adjoint is not linear, the Linear Operator FTT with real=True does not
    pass the dot-test.

    """
    def __init__(self, dims, dir=0, nfft=None, sampling=1., real=False, dtype='complex64'):
        if isinstance(dims, int):
            dims = (dims,)
        if dir > len(dims)-1:
            raise ValueError('dir=%d must be smaller than '
                             'number of dims=%d...' % (dir, len(dims)))
        self.dir = dir
        self.nfft = nfft if nfft is not None else dims[self.dir]
        self.real = real
        self.f = np.fft.rfftfreq(self.nfft, d=sampling) if real \
                 else np.fft.fftfreq(self.nfft, d=sampling)
        if len(dims) == 1:
            self.dims = np.array([dims[0], 1])
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft
            self.reshape = False
        else:
            self.dims = np.array(dims)
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft
            self.reshape = True
        self.shape = (int(np.prod(dims)*(self.nfft//2 + 1 if self.real else self.nfft)/self.dims[dir]),
                      int(np.prod(dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            if self.real:
                y = np.sqrt(1./self.nfft)*np.fft.rfft(x, n=self.nfft, axis=-1)
            else:
                y = np.sqrt(1./self.nfft)*np.fft.fft(x, n=self.nfft, axis=-1)
        else:
            x = np.reshape(x, self.dims)
            if self.dir < len(self.dims)-1: # need to bring the dimension to transform to last dimension
                x = np.swapaxes(x, self.dir, -1)
            if self.real:
                y = np.sqrt(1. / self.nfft) * np.fft.rfft(x, n=self.nfft, axis=-1)
            else:
                y = np.sqrt(1. / self.nfft) * np.fft.fft(x, n=self.nfft, axis=-1)
            if self.dir < len(self.dims)-1:
                y = np.swapaxes(y, -1, self.dir)
            y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            if self.real:
                y = np.sqrt(self.nfft)*np.fft.irfft(x, n=self.nfft, axis=-1)
            else:
                y = np.sqrt(self.nfft)*np.fft.ifft(x, n=self.nfft, axis=-1)
            if self.nfft != self.dims[self.dir]:
                y = y[:self.dims[self.dir]]
        else:
            x = np.reshape(x, self.dims_fft)
            if self.dir < len(self.dims)-1: # need to bring the dimension to transform to last dimension
                x = np.swapaxes(x, self.dir, -1)
            if self.real:
                y = np.sqrt(self.nfft) * np.fft.irfft(x, n=self.nfft, axis=-1)
            else:
                y = np.sqrt(self.nfft) * np.fft.ifft(x, n=self.nfft, axis=-1)
            if self.nfft != self.dims[self.dir]:
                y = y[..., :self.dims[self.dir]]
            if self.dir < len(self.dims)-1:
                y = np.swapaxes(y, -1, self.dir)
            y = np.ndarray.flatten(y)
        return y