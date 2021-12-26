import logging
import warnings
import numpy as np
from pylops import LinearOperator

try:
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None
    pyfftw_message = 'Pyfftw not installed, use numpy or run ' \
                     '"pip install pyFFTW" or ' \
                     '"conda install -c conda-forge pyfftw".'
except Exception as e:
    pyfftw = None
    pyfftw_message = 'Failed to import pyfftw (error:%s), use numpy.' % e

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class _FFT_numpy(LinearOperator):
    """One dimensional Fast-Fourier Transform using numpy"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        real=False,
        fftshift=None,
        ifftshift_before=None,
        fftshift_after=False,
        dtype="complex128",
    ):
        if isinstance(dims, int):
            dims = (dims,)
        if dir > len(dims)-1:
            raise ValueError('dir=%d must be smaller than '
                             'number of dims=%d...' % (dir, len(dims)))
        self.dir = dir
        self.nfft = nfft if nfft is not None else dims[self.dir]
        self.real = real

        # Use fftshift if supplied, otherwise use ifftshift_before
        # If neither are supplied, set to False
        if fftshift is not None:
            warnings.warn(
                "fftshift is deprecated. Please use ifftshift_before.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            if ifftshift_before is not None:
                warnings.warn(
                    "Passed fftshift and ifftshift_before, ignoring ifftshift_before. ",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            ifftshift_before = fftshift
        if fftshift is None and ifftshift_before is None:
            ifftshift_before = False
        self.ifftshift_before = ifftshift_before

        self.f = (
            np.fft.rfftfreq(self.nfft, d=sampling)
            if real
                 else np.fft.fftfreq(self.nfft, d=sampling)
        )
        self.fftshift_after = fftshift_after
        if self.fftshift_after:
            if self.real:
                warnings.warn(
                    "Using fftshift_after with real=True. fftshift should only be applied after a complex FFT. This is rarely intended behavior but if it is, ignore this message."
                )
            self.f = np.fft.fftshift(self.f)

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
        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
        # Find types to enforce to forward and adjoint outputs. This is
        # required as np.fft.fft always returns complex128 even if input is
        # float32 or less. Moreover, when choosing real=True, the type of the
        # adjoint output is forced to be real even if the provided dtype
        # is complex.
        self.rdtype = np.real(np.ones(1, dtype)).dtype if real else np.dtype(dtype)
        self.cdtype = (np.ones(1, dtype=self.rdtype) +
                       1j * np.ones(1, dtype=self.rdtype)).dtype
        self.dtype = self.cdtype
        self.clinear = False if real else True
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            x = x.ravel()
            if self.ifftshift_before:
                x = np.fft.ifftshift(x)
            if self.real:
                y = np.fft.rfft(np.real(x), n=self.nfft, axis=-1, norm='ortho')
                # Apply scaling to obtain a correct adjoint for this operator
                y[..., 1:1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            else:
                y = np.fft.fft(x, n=self.nfft, axis=-1, norm='ortho')
            if self.fftshift_after:
                y = np.fft.fftshift(y)
        else:
            x = np.reshape(x, self.dims)
            if self.ifftshift_before:
                x = np.fft.ifftshift(x, axes=self.dir)
            if self.real:
                y = np.fft.rfft(np.real(x), n=self.nfft,
                                axis=self.dir, norm='ortho')
                # Apply scaling to obtain a correct adjoint for this operator
                y = np.swapaxes(y, -1, self.dir)
                y[..., 1:1 + (self.nfft - 1) // 2] *= np.sqrt(2)
                y = np.swapaxes(y, self.dir, -1)
            else:
                y = np.fft.fft(x, n=self.nfft,
                               axis=self.dir, norm='ortho')
            if self.fftshift_after:
                y = np.fft.fftshift(y, axes=self.dir)
            y = y.flatten()
        y = y.astype(self.cdtype)
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            x = x.ravel()
            if self.fftshift_after:
                x = np.fft.ifftshift(x)
            if self.real:
                # Apply scaling to obtain a correct adjoint for this operator
                x = x.copy()
                x[..., 1:1 + (self.nfft - 1) // 2] /= np.sqrt(2)
                y = np.fft.irfft(x, n=self.nfft, axis=-1, norm='ortho')
            else:
                y = np.fft.ifft(x, n=self.nfft, axis=-1, norm='ortho')
            if self.nfft != self.dims[self.dir]:
                y = y[:self.dims[self.dir]]
            if self.ifftshift_before:
                y = np.fft.fftshift(y)
        else:
            x = np.reshape(x, self.dims_fft)
            if self.fftshift_after:
                x = np.fft.ifftshift(x, axes=self.dir)
            if self.real:
                # Apply scaling to obtain a correct adjoint for this operator
                x = x.copy()
                x = np.swapaxes(x, -1, self.dir)
                x[..., 1:1 + (self.nfft - 1) // 2] /= np.sqrt(2)
                x = np.swapaxes(x, self.dir, -1)
                y = np.fft.irfft(x, n=self.nfft, axis=self.dir, norm='ortho')
            else:
                y = np.fft.ifft(x, n=self.nfft, axis=self.dir, norm='ortho')
            if self.nfft != self.dims[self.dir]:
                y = np.take(y, np.arange(0, self.dims[self.dir]),
                            axis=self.dir)
            if self.ifftshift_before:
                y = np.fft.fftshift(y, axes=self.dir)
            y = y.flatten()
        y = y.astype(self.rdtype)
        return y


class _FFT_fftw(LinearOperator):
    """One dimensional Fast-Fourier Transform using pyffw"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        real=False,
        fftshift=None,
        ifftshift_before=None,
        fftshift_after=False,
        dtype="complex128",
        **kwargs_fftw
    ):
        if isinstance(dims, int):
            dims = (dims,)
        if dir > len(dims)-1:
            raise ValueError('dir=%d must be smaller than '
                             'number of dims=%d...' % (dir, len(dims)))
        self.dir = dir
        if nfft is None:
            nfft = dims[self.dir]
        elif nfft < dims[self.dir]:
            logging.warning('nfft should be bigger or equal then '
                            ' dims[self.dir] for engine=fftw, set to '
                            'dims[self.dir]')
            nfft = dims[self.dir]
        self.nfft = nfft

        self.real = real

        # Use fftshift if supplied, otherwise use ifftshift_before
        # If neither are supplied, set to False
        if fftshift is not None:
            warnings.warn(
                "fftshift is deprecated. Please use ifftshift_before.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            if ifftshift_before is not None:
                warnings.warn(
                    "Passed fftshift and ifftshift_before, ignoring ifftshift_before. ",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            ifftshift_before = fftshift
        if fftshift is None and ifftshift_before is None:
            ifftshift_before = False
        self.ifftshift_before = ifftshift_before

        self.f = (
            np.fft.rfftfreq(self.nfft, d=sampling)
            if real
                 else np.fft.fftfreq(self.nfft, d=sampling)
        )
        self.fftshift_after = fftshift_after
        if self.fftshift_after:
            if self.real:
                warnings.warn(
                    "Using fftshift_after with real=True. fftshift should only be applied after a complex FFT. This is rarely intended behavior but if it is, ignore this message."
                )
            self.f = np.fft.fftshift(self.f)
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
        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
        self.rdtype = np.real(np.ones(1, dtype)).dtype if real else np.dtype(dtype)
        self.cdtype = (np.ones(1, dtype=self.rdtype) +
                       1j * np.ones(1, dtype=self.rdtype)).dtype
        self.dtype = self.cdtype
        self.clinear = False if real else True
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
        self.x = pyfftw.empty_aligned(self.dims_t,
                                      dtype=self.rdtype if real else self.cdtype)
        self.y = pyfftw.empty_aligned(self.dims_fft,
                                      dtype=self.cdtype)
        self.fftplan = pyfftw.FFTW(self.x, self.y, axes=(self.dir,),
                                   direction='FFTW_FORWARD', **kwargs_fftw)
        self.ifftplan = pyfftw.FFTW(self.y, self.x, axes=(self.dir,),
                                    direction='FFTW_BACKWARD', **kwargs_fftw)

    def _matvec(self, x):
        if self.real:
            x = np.real(x)
        if not self.reshape:
            x = x.ravel()
            if self.ifftshift_before:
                x = np.fft.ifftshift(x)
            if self.dopad:
                x = np.pad(x, self.pad, 'constant', constant_values=0)
            y = np.sqrt(1. / self.nfft) * self.fftplan(x)
            if self.real:
                y[..., 1:1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            if self.fftshift_after:
                y = np.fft.fftshift(y)
        else:
            x = np.reshape(x, self.dims)
            if self.ifftshift_before:
                x = np.fft.ifftshift(x, axes=self.dir)
            if self.dopad:
                x = np.pad(x, self.pad, 'constant', constant_values=0)
            y = np.sqrt(1. / self.nfft) * self.fftplan(x)
            if self.real:
                # Apply scaling to obtain a correct adjoint for this operator
                y = np.swapaxes(y, -1, self.dir)
                y[..., 1:1 + (self.nfft - 1) // 2] *= np.sqrt(2)
                y = np.swapaxes(y, self.dir, -1)
            if self.fftshift_after:
                y = np.fft.fftshift(y, axes=self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        if not self.reshape:
            x = x.ravel()
            if self.fftshift_after:
                x = np.fft.ifftshift(x)
            if self.real:
                # Apply scaling to obtain a correct adjoint for this operator
                x = x.copy()
                x[..., 1:1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            y = np.sqrt(self.nfft) * self.ifftplan(x)
            if self.nfft != self.dims[self.dir]:
                y = y[:self.dims[self.dir]]
            if self.ifftshift_before:
                y = np.fft.fftshift(y)
        else:
            x = np.reshape(x, self.dims_fft)
            if self.fftshift_after:
                x = np.fft.ifftshift(x, axes=self.dir)
            if self.real:
                # Apply scaling to obtain a correct adjoint for this operator
                x = x.copy()
                x = np.swapaxes(x, -1, self.dir)
                x[..., 1:1 + (self.nfft - 1) // 2] /= np.sqrt(2)
                x = np.swapaxes(x, self.dir, -1)
            y = np.sqrt(self.nfft) * self.ifftplan(x)
            if self.nfft != self.dims[self.dir]:
                y = np.take(y, np.arange(0, self.dims[self.dir]),
                            axis=self.dir)
            if self.ifftshift_before:
                y = np.fft.fftshift(y, axes=self.dir)
        if self.real:
            y = np.real(y)
        return y.ravel()


def FFT(
    dims,
    dir=0,
    nfft=None,
    sampling=1.0,
    real=False,
    fftshift=None,
    ifftshift_before=None,
    fftshift_after=False,
    engine="numpy",
    dtype="complex128",
    **kwargs_fftw
):
    r"""One dimensional Fast-Fourier Transform.

    Apply Fast-Fourier Transform (FFT) along a specific direction ``dir`` of a
    multi-dimensional array of size ``dim``.

    Note that the FFT operator is an overload to either the numpy
    :py:func:`numpy.fft.fft` (or :py:func:`numpy.fft.rfft` for real models) in
    forward mode and to the numpy :py:func:`numpy.fft.ifft` (or
    :py:func:`numpy.fft.irfft` for real models) in adjoint mode, or their cupy
    equivalents. Alternatively, the :py:class:`pyfftw.FFTW` class is used
    when ``engine='fftw'`` is chosen.

    In both cases, scaling is properly taken into account to guarantee
    that the operator is passing the dot-test. If a user is interested to use
    the unscaled forward FFT, it must pre-multiply the operator by an
    appropriate correction factor. Moreover, for a real valued
    input signal, it is advised to use the flag `real=True` as it stores
    the values of the Fourier transform at positive frequencies only as
    values at negative frequencies are simply their complex conjugates.

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
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    fftshift : :obj:`bool`, optional
        Apply ifftshift/fftshift (``True``) or not (``False``) to model vector.
        This is required when the model is arranged over a symmetric time axis
        such that it is first rearranged before applying the Fourier Transform.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``). Choose
        ``numpy`` when working with cupy arrays.
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the `dtype` of the operator
        is the corresponding complex type even when a real type is provided.
        Nevertheless, the provided dtype will be enforced on the vector
        returned by the `rmatvec` method.
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
        If ``dims`` is provided and ``dir`` is bigger than ``len(dims)``
    NotImplementedError
        If ``engine`` is neither ``numpy`` nor ``fftw``

    Notes
    -----
    The FFT operator applies the forward Fourier transform to a signal
    :math:`d(t)` in forward mode:

    .. math::
        D(f) = \mathscr{F} (d) = \frac{1}{\sqrt{N_F}} \int d(t) e^{-j2\pi ft} dt

    Similarly, the inverse Fourier transform is applied to the Fourier spectrum
    :math:`D(f)` in adjoint mode:

    .. math::
        d(t) = \mathscr{F}^{-1} (D) = \sqrt{N_F} \int D(f) e^{j2\pi ft} df

    where :math:`N_F` is the number of samples in the Fourier domain `nfft`.
    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform. Note that the FFT operator is a
    special operator in that the adjoint is also the inverse of the forward mode.
    Moreover, in case of real signal in time domain, the Fourier transform in
    Hermitian.

    """
    if engine == 'fftw' and pyfftw is not None:
        f = _FFT_fftw(
            dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            fftshift=fftshift,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            **kwargs_fftw
        )
    elif engine == 'numpy' or (engine == 'fftw' and pyfftw is None):
        if engine == 'fftw' and pyfftw is None:
            logging.warning(pyfftw_message)
        f = _FFT_numpy(
            dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            fftshift=fftshift,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    else:
        raise NotImplementedError('engine must be numpy or fftw')
    return f
