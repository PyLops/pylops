import logging
import numpy as np
from pylops import LinearOperator

try:
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class _FFT_numpy(LinearOperator):
    """One dimensional Fast-Fourier Transform using numpy
    """
    def __init__(self, dims, dir=0, nfft=None, sampling=1.,
                 real=False, dtype='complex128'):
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
            self.dims_fft[self.dir] = self.nfft // 2 + 1 if \
                self.real else self.nfft
            self.reshape = False
        else:
            self.dims = np.array(dims)
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft // 2 + 1 if \
                self.real else self.nfft
            self.reshape = True
        self.shape = (int(np.prod(dims)*(self.nfft//2 + 1 if self.real
                                         else self.nfft)/self.dims[dir]),
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
            if self.real:
                y = np.sqrt(1. / self.nfft) * np.fft.rfft(x, n=self.nfft,
                                                          axis=self.dir)
            else:
                y = np.sqrt(1. / self.nfft) * np.fft.fft(x, n=self.nfft,
                                                         axis=self.dir)
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
            if self.real:
                y = np.sqrt(self.nfft) * np.fft.irfft(x, n=self.nfft,
                                                      axis=self.dir)
            else:
                y = np.sqrt(self.nfft) * np.fft.ifft(x, n=self.nfft,
                                                     axis=self.dir)
            if self.nfft != self.dims[self.dir]:
                y = np.take(y, np.arange(0, self.dims[self.dir]),
                            axis=self.dir)
            y = np.ndarray.flatten(y)
        return y


class _FFT_fftw(LinearOperator):
    """One dimensional Fast-Fourier Transform using pyffw
    """
    def __init__(self, dims, dir=0, nfft=None, sampling=1.,
                 real=False, dtype='complex128', **kwargs_fftw):
        if isinstance(dims, int):
            dims = (dims,)
        if dir > len(dims)-1:
            raise ValueError('dir=%d must be smaller than '
                             'number of dims=%d...' % (dir, len(dims)))
        self.dir = dir
        if nfft is not None and nfft >= dims[self.dir]:
            self.nfft = nfft
        else:
            logging.warning('nfft should be bigger or equal then dims[self.dir]'
                            ' for engine=fftw, set to dims[self.dir]')
            self.nfft = dims[self.dir]
        self.real = real
        self.f = np.fft.rfftfreq(self.nfft, d=sampling) if real \
                 else np.fft.fftfreq(self.nfft, d=sampling)
        if len(dims) == 1:
            self.dims = np.array([dims[0], ])
            self.dims_t = self.dims.copy()
            self.dims_t[self.dir] = self.nfft
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft // 2 + 1 if \
                self.real else self.nfft
            self.reshape = False
        else:
            self.dims = np.array(dims)
            self.dims_t = self.dims.copy()
            self.dims_t[self.dir] = self.nfft
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft//2 + 1 if \
                self.real else self.nfft
            self.reshape = True
        self.shape = (int(np.prod(dims)*(self.nfft//2 + 1 if self.real
                                         else self.nfft)/self.dims[dir]),
                      int(np.prod(dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # define padding(fftw requires the user to provide padded input signal)
        self.pad = [[0, 0] for _ in range(len(self.dims))]
        if self.real:
            if self.nfft % 2:
                self.pad[self.dir][1] = 2*(self.dims_fft[self.dir]-1) + 1 - \
                                        self.dims[self.dir]
            else:
                self.pad[self.dir][1] = 2*(self.dims_fft[self.dir]-1) - \
                                        self.dims[self.dir]
        else:
            self.pad[self.dir][1] = self.dims_fft[self.dir] - \
                                      self.dims[self.dir]
        self.dopad = True if np.sum(np.array(self.pad)) > 0 else False

        # create empty arrays and plans for fft/ifft
        xtype = np.real(np.ones(1, dtype=self.dtype)).dtype # find model type
        self.x = pyfftw.empty_aligned(self.dims_t,
                                      dtype=xtype if real else self.dtype)
        self.y = pyfftw.empty_aligned(self.dims_fft,
                                      dtype=self.dtype)
        self.fftplan = pyfftw.FFTW(self.x, self.y, axes=(self.dir,),
                                   direction='FFTW_FORWARD', **kwargs_fftw)
        self.ifftplan = pyfftw.FFTW(self.y, self.x, axes=(self.dir,),
                                    direction='FFTW_BACKWARD', **kwargs_fftw)

    def _matvec(self, x):
        if not self.reshape:
            if self.dopad:
                x = np.pad(x, self.pad, 'constant', constant_values=0)
            y = np.sqrt(1./self.nfft)*self.fftplan(x)
        else:
            x = np.reshape(x, self.dims)
            if self.dopad:
                x = np.pad(x, self.pad, 'constant', constant_values=0)
            y = np.sqrt(1. / self.nfft) * self.fftplan(x)
            y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            y = np.sqrt(self.nfft) * self.ifftplan(x)
            if self.nfft != self.dims[self.dir]:
                y = y[:self.dims[self.dir]]
        else:
            x = np.reshape(x, self.dims_fft)
            y = np.sqrt(self.nfft) * self.ifftplan(x)
            if self.nfft != self.dims[self.dir]:
                y = np.take(y, np.arange(0, self.dims[self.dir]),
                            axis=self.dir)
            y = np.ndarray.flatten(y)
        return y


def FFT(dims, dir=0, nfft=None, sampling=1.,
        real=False, engine='numpy', dtype='complex128', **kwargs_fftw):
    r"""One dimensional Fast-Fourier Transform.

    Apply Fast-Fourier Transform (FFT) along a specific direction ``dir`` of a
    multi-dimensional array of size ``dim``.

    Note that the FFT operator is an overload to either the numpy
    :py:func:`numpy.fft.fft` (or :py:func:`numpy.fft.rfft` for real models) in
    forward mode and to the numpy :py:func:`numpy.fft.ifft` (or
    :py:func:`numpy.fft.irfft` for real models) in adjoint mode, or to the
    :py:class:`pyfftw.FFTW` class.

    In both cases, scaling is properly taken into account to guarantee
    that the operator is passing the dot-test.

    .. note:: For a real valued input signal, it is possible to store the
      values of the Fourier transform at positive frequencies only as values
      at negative frequencies are simply their complex conjugates.
      However as the operation of removing the negative part of the frequency
      axis in forward mode and adding the complex conjugates in adjoint mode
      is nonlinear, the Linear Operator FTT with ``real=True`` is not expected
      to pass the dot-test. It is thus *only* advised to use this flag when a
      forward and adjoint FFT is used in the same chained operator
      (e.g., ``FFT.H*Op*FFT``) such as in
      :py:func:`pylops.waveeqprocessing.mdd.MDC`.

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
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    **kwargs_fftw
            Arbitrary keyword arguments
            for :py:class:`pyfftw.FTTW`

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    Raises
    ------
    ValueError
        If ``dims`` is not provided and if ``dir`` is bigger than ``len(dims)``

    Notes
    -----
    The FFT operator applies the forward Fourier transform to a signal
    :math:`d(t)` in forward mode:

    .. math::
        D(f) = \mathscr{F} (d) = \int d(t) e^{-j2\pi ft} dt

    Similarly, the inverse Fourier transform is applied to the Fourier spectrum
    :math:`D(f)` in adjoint mode:

    .. math::
        d(t) = \mathscr{F}^{-1} (D) = \int D(f) e^{j2\pi ft} df

    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform.

    Note that the FFT operator is a special operator in that the adjoint is
    also the inverse of the forward mode. Moreover, in case of real signal
    in time domain, the Fourier transform in Hermitian.

    """
    if engine == 'fftw' and pyfftw is not None:
        f = _FFT_fftw(dims, dir=dir, nfft=nfft,
                      sampling=sampling, real=real,
                      dtype=dtype, **kwargs_fftw)
    elif engine == 'numpy' or pyfftw is None:
        if pyfftw is None:
            logging.warning('use numpy, pyfftw not available...')
        f = _FFT_numpy(dims, dir=dir, nfft=nfft,
                       sampling=sampling, real=real, dtype=dtype)
    else:
        raise ValueError('engine must be numpy or fftw')

    return f
