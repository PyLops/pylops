import logging
import warnings

import numpy as np
import scipy.fft

from pylops.signalprocessing._BaseFFTs import _BaseFFT

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
        norm="ortho",
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
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        if self.cdtype != np.complex128:
            warnings.warn(
                f"numpy backend always returns complex128 dtype. To respect the passed dtype, data will be casted to {self.cdtype}."
            )
        # FFTs are called with "ortho" for backwards compatibility
        # The factors below are conversions factors ortho->norm
        if self.norm == "backward":
            self._scale = np.sqrt(self.nfft)
        elif self.norm == "forward":
            self._scale = np.sqrt(1.0 / self.nfft)

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
        if self.norm != "ortho":
            y *= self._scale
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
        if self.norm != "ortho":
            y *= self._scale
        if self.nfft != self.dims[self.dir]:
            y = np.take(y, np.arange(0, self.dims[self.dir]), axis=self.dir)
        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before:
            y = np.fft.fftshift(y, axes=self.dir)
        y = y.ravel()
        y = y.astype(self.rdtype)
        return y


class _FFT_scipy(_BaseFFT):
    """One dimensional Fast-Fourier Transform using numpy"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        norm="ortho",
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
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        # FFTs are called with "ortho" for backwards compatibility
        # The factors below are conversions factors ortho->norm
        if self.norm == "backward":
            self._scale = np.sqrt(self.nfft)
        elif self.norm == "forward":
            self._scale = np.sqrt(1.0 / self.nfft)

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before:
            x = scipy.fft.ifftshift(x, axes=self.dir)
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = scipy.fft.rfft(x, n=self.nfft, axis=self.dir, norm="ortho")
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dir)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dir, -1)
        else:
            y = scipy.fft.fft(x, n=self.nfft, axis=self.dir, norm="ortho")
        if self.norm != "ortho":
            y *= self._scale
        if self.fftshift_after:
            y = scipy.fft.fftshift(y, axes=self.dir)
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after:
            x = scipy.fft.ifftshift(x, axes=self.dir)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dir)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dir, -1)
            y = scipy.fft.irfft(x, n=self.nfft, axis=self.dir, norm="ortho")
        else:
            y = scipy.fft.ifft(x, n=self.nfft, axis=self.dir, norm="ortho")
        if self.norm != "ortho":
            y *= self._scale
        if self.nfft != self.dims[self.dir]:
            y = np.take(y, np.arange(0, self.dims[self.dir]), axis=self.dir)
        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before:
            y = scipy.fft.fftshift(y, axes=self.dir)
        y = y.ravel()
        return y


class _FFT_fftw(_BaseFFT):
    """One dimensional Fast-Fourier Transform using pyffw"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        norm="ortho",
        real=False,
        ifftshift_before=None,
        fftshift_after=False,
        dtype="complex128",
        **kwargs_fftw,
    ):
        if np.dtype(dtype) == np.float16:
            warnings.warn(
                "fftw backend is unavailable with float16 dtype. Will use float32."
            )
            dtype = np.float32
        super().__init__(
            dims=dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        if self.cdtype != np.complex128:
            warnings.warn(
                f"fftw backend returns complex128 dtype. To respect the passed dtype, data will be casted to {self.cdtype}."
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

        if "ortho" in kwargs_fftw:
            warnings.warn(
                f"FFTW option 'ortho' will be overwritten by norm={self.norm}"
            )
        if "normalise_idft" in kwargs_fftw:
            warnings.warn(
                f"FFTW option 'normalise_idft' will be overwritten by norm={self.norm}"
            )
        if self.norm == "ortho":
            kwargs_fftw["ortho"] = True
            kwargs_fftw["normalise_idft"] = False
        elif self.norm == "backward":
            self._scale = self.nfft
            kwargs_fftw["ortho"] = False
            kwargs_fftw["normalise_idft"] = False
        elif self.norm == "forward":
            self._scale = 1.0 / self.nfft
            kwargs_fftw["ortho"] = False
            kwargs_fftw["normalise_idft"] = True
        else:
            raise ValueError(
                f"'{self.norm}' is not one of 'ortho', 'backward' or 'forward'"
            )
        self.fftplan = pyfftw.FFTW(
            self.x, self.y, axes=(self.dir,), direction="FFTW_FORWARD", **kwargs_fftw
        )
        self.ifftplan = pyfftw.FFTW(
            self.y, self.x, axes=(self.dir,), direction="FFTW_BACKWARD", **kwargs_fftw
        )

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before:
            x = np.fft.ifftshift(x, axes=self.dir)
        if not self.clinear:
            x = np.real(x)
        if self.dopad:
            x = np.pad(x, self.pad, "constant", constant_values=0)

        # self.fftplan() always uses byte-alligned self.x as input array and
        # returns self.y as output array. As such, self.y must be copied so as
        # not to be overwritten on a subsequent call to _matvec.
        np.copyto(self.x, x)
        y = self.fftplan().copy()
        if self.norm == "forward":
            y *= self._scale

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

        # self.ifftplan() always uses byte-alligned self.y as input array.
        # We copy here so we don't need to copy again in the case of `real=True`,
        # which only performs operations that preserve byte-allignment.
        np.copyto(self.y, x)
        x = self.y  # Update reference only

        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = np.swapaxes(x, -1, self.dir)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dir, -1)

        # self.ifftplan() always returns self.x, which must be copied so as not
        # to be overwritten on a subsequent call to _rmatvec.
        y = self.ifftplan().copy()

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
    norm="ortho",
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

    Using the default NumPy engine, the FFT operator is an overload to either the NumPy
    :py:func:`numpy.fft.fft` (or :py:func:`numpy.fft.rfft` for real models) in
    forward mode, and to :py:func:`numpy.fft.ifft` (or :py:func:`numpy.fft.irfft`
    for real models) in adjoint mode, or their CuPy equivalents.
    When ``engine='fftw'`` is chosen, the :py:class:`pyfftw.FFTW` class is used
    instead.
    Alternatively, when the SciPy engine is chosen, the overloads are of
    :py:func:`scipy.fft.fft` (or :py:func:`scipy.fft.rfft` for real models) in
    forward mode, and to :py:func:`scipy.fft.ifft` (or :py:func:`scipy.fft.irfft`
    for real models) in adjoint mode.

    When using `real=True`, the result of the forward is also multiplied by sqrt(2)
    for all frequency bins except zero and Nyquist, and the input of the adjoint is
    divided by sqrt(2) for the same frequencies.

    For a real valued input signal, it is advised to use the flag `real=True`
    as it stores the values of the Fourier transform at positive frequencies only as
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
    norm : `{"ortho", "backward", "forward"}`, optional
        Normalization mode (see :py:func:`numpy.fft.fft`). Note that for "backward"
        and "forward", the scaling placed on the forward is the same as that placed
        on the adjoint, so as to respect adjoitness. This is different from standard
        NumPy/SciPy behavior which scales ``fft`` and ``ifft`` differently when using
        the same ``norm``. As a result, a forward and adjoint pass with the "backward"
        norm will introduce a factor of :math:`nfft`; a forward and adjoint pass with
        "forward" will introduce a factor of :math:`nfft^{-1}`. Only "ortho" will
        recover the original signal.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    fftshift : :obj:`bool`, optional
        Note: ``fftshift`` is deprecated, use ``ifftshift_before``.
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
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``fftw``, or ``scipy``). Choose
        ``numpy`` when working with cupy arrays.
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the ``dtype`` of the operator
        is the corresponding complex type even when a real type is provided.
        In addition, note that neither the NumPy nor the FFTW backends supports
        returning ``dtype``s different than ``complex128``. As such, when using either
        backend, arrays will be force-casted to types corresponding to the supplied ``dtype``.
        The SciPy backend supports all precisions natively.
        Under all backends, when a real ``dtype`` is supplied, a real result will be
        enforced on the result of the ``rmatvec`` and the input of the ``matvec``.
    **kwargs_fftw
            Arbitrary keyword arguments
            for :py:class:`pyfftw.FTTW`

    Attributes
    ----------
    dims_fft : :obj:`tuple`
        Shape of the array after the forward, but before linearization. E.g.
        ``y_reshaped = (Op * x.ravel()).reshape(Op.dims_fft)``.
    f : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies
    real : :obj:`bool`
        When True, uses ``rfft``/``irfft``
    rdtype : :obj:`bool`
        Expected input type to the forward
    cdtype : :obj:`bool`
        Output type of the forward. Complex equivalent to ``rdtype``.
    shape : :obj:`tuple`
        Operator shape
    clinear : :obj:`bool`
        Operator is complex-linear. Is false when either ``real=True`` or when
        ``dtype`` is not a complex type.
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    Raises
    ------
    ValueError
        If ``dims`` is provided and ``dir`` is bigger than ``len(dims)``
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``fftw``, nor ``scipy``.

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
            norm=norm,
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
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    elif engine == "scipy":
        f = _FFT_scipy(
            dims,
            dir=dir,
            nfft=nfft,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    else:
        raise NotImplementedError("engine must be numpy, fftw or scipy")
    return f
