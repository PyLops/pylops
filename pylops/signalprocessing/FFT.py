import logging
import warnings

import numpy as np

from ._BaseFFTs import _BaseFFT

try:
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None
    pyfftw_message = (
        "Pyfftw not installed, use numpy or run "
        '"pip install pyFFTW" or '
        '"conda install -c conda-forge pyfftw".'
    )
except Exception as e:
    pyfftw = None
    pyfftw_message = "Failed to import pyfftw (error:%s), use numpy." % e

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class _FFT_numpy(_BaseFFT):
    """One dimensional Fast-Fourier Transform using numpy"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        super().__init__(
            dims=dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before:
            x = np.fft.ifftshift(x, axes=self.dir)
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = np.fft.rfft(x, n=self.nfft, axis=self.dir, norm="ortho")
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dir)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dir, -1)
        else:
            y = np.fft.fft(x, n=self.nfft, axis=self.dir, norm="ortho")
        if self.fftshift_after:
            y = np.fft.fftshift(y, axes=self.dir)
        y = y.ravel()
        y = y.astype(self.cdtype)
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after:
            x = np.fft.ifftshift(x, axes=self.dir)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dir)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dir, -1)
            y = np.fft.irfft(x, n=self.nfft, axis=self.dir, norm="ortho")
        else:
            y = np.fft.ifft(x, n=self.nfft, axis=self.dir, norm="ortho")
        if self.nfft != self.dims[self.dir]:
            y = np.take(y, np.arange(0, self.dims[self.dir]), axis=self.dir)
        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before:
            y = np.fft.fftshift(y, axes=self.dir)
        y = y.ravel()
        y = y.astype(self.rdtype)
        return y


class _FFT_fftw(_BaseFFT):
    """One dimensional Fast-Fourier Transform using pyffw"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        real=False,
        ifftshift_before=None,
        fftshift_after=False,
        dtype="complex128",
        **kwargs_fftw,
    ):
        super().__init__(
            dims=dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        self.dims_t = self.dims.copy()
        self.dims_t[self.dir] = self.nfft

        # define padding(fftw requires the user to provide padded input signal)
        self.pad = np.zeros((self.ndim, 2), dtype=int)
        if self.real:
            if self.nfft % 2:
                self.pad[self.dir, 1] = (
                    2 * (self.dims_fft[self.dir] - 1) + 1 - self.dims[self.dir]
                )
            else:
                self.pad[self.dir, 1] = (
                    2 * (self.dims_fft[self.dir] - 1) - self.dims[self.dir]
                )
        else:
            self.pad[self.dir, 1] = self.dims_fft[self.dir] - self.dims[self.dir]
        self.dopad = True if np.sum(self.pad) > 0 else False

        # create empty arrays and plans for fft/ifft
        self.x = pyfftw.empty_aligned(
            self.dims_t, dtype=self.rdtype if real else self.cdtype
        )
        self.y = pyfftw.empty_aligned(self.dims_fft, dtype=self.cdtype)
        self.fftplan = pyfftw.FFTW(
            self.x, self.y, axes=(self.dir,), direction="FFTW_FORWARD", **kwargs_fftw
        )
        self.ifftplan = pyfftw.FFTW(
            self.y, self.x, axes=(self.dir,), direction="FFTW_BACKWARD", **kwargs_fftw
        )

    def _matvec(self, x):
        if self.real:
            x = np.real(x)
        x = np.reshape(x, self.dims)
        if self.ifftshift_before:
            x = np.fft.ifftshift(x, axes=self.dir)
        if not self.clinear:
            x = np.real(x)
        if self.dopad:
            x = np.pad(x, self.pad, "constant", constant_values=0)
        y = np.sqrt(1.0 / self.nfft) * self.fftplan(x)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dir)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dir, -1)
        if self.fftshift_after:
            y = np.fft.fftshift(y, axes=self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after:
            x = np.fft.ifftshift(x, axes=self.dir)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dir)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dir, -1)
        y = np.sqrt(self.nfft) * self.ifftplan(x)
        if self.nfft != self.dims[self.dir]:
            y = np.take(y, np.arange(0, self.dims[self.dir]), axis=self.dir)
        if self.ifftshift_before:
            y = np.fft.fftshift(y, axes=self.dir)
        if not self.clinear:
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
    **kwargs_fftw,
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
        Note: `fftshift` is deprecated, use `ifftshift_before`.
    ifftshift_before : :obj:`bool`, optional
        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform.
        Defaults to not applying ifftshift.
    fftshift_after : :obj:`bool`, optional
        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero.
        Defaults to not applying fftshift.
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
    clinear : :obj:`bool`
        Operator is complex-linear. Is false when either real=True or when
        dtype is not a complex type.
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

    if engine == "fftw" and pyfftw is not None:
        f = _FFT_fftw(
            dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            **kwargs_fftw,
        )
    elif engine == "numpy" or (engine == "fftw" and pyfftw is None):
        if engine == "fftw" and pyfftw is None:
            logging.warning(pyfftw_message)
        f = _FFT_numpy(
            dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    else:
        raise NotImplementedError("engine must be numpy or fftw")
    return f
