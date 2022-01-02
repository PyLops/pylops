import logging
import warnings

import numpy as np
import scipy.fft

from pylops.signalprocessing._BaseFFTs import _BaseFFTND

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class _FFT2D_numpy(_BaseFFTND):
    """Two dimensional Fast-Fourier Transform using numpy"""

    def __init__(
        self,
        dims,
        dirs=(0, 1),
        nffts=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        super().__init__(
            dims=dims,
            dirs=dirs,
            nffts=nffts,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        if self.cdtype != np.complex128:
            warnings.warn(
                f"numpy backend always returns complex128 dtype. To respect the passed dtype, data will be casted to {self.cdtype}."
            )

        # checks
        if self.ndim < 2:
            raise ValueError("FFT2D requires at least two input dimensions")
        if self.ndirs != 2:
            raise ValueError("FFT2D must be applied along exactly two dimensions")

        self.f1, self.f2 = self.fs
        del self.fs

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before.any():
            x = np.fft.ifftshift(x, axes=self.dirs[self.ifftshift_before])
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = np.fft.rfft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dirs[-1])
            y[..., 1 : 1 + (self.nffts[-1] - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dirs[-1], -1)
        else:
            y = np.fft.fft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        y = y.astype(self.cdtype)
        if self.fftshift_after.any():
            y = np.fft.fftshift(y, axes=self.dirs[self.fftshift_after])
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after.any():
            x = np.fft.ifftshift(x, axes=self.dirs[self.fftshift_after])
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dirs[-1])
            x[..., 1 : 1 + (self.nffts[-1] - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dirs[-1], -1)
            y = np.fft.irfft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        else:
            y = np.fft.ifft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        y = np.take(y, range(self.dims[self.dirs[0]]), axis=self.dirs[0])
        y = np.take(y, range(self.dims[self.dirs[1]]), axis=self.dirs[1])
        if not self.clinear:
            y = np.real(y)
        y = y.astype(self.rdtype)
        if self.ifftshift_before.any():
            y = np.fft.fftshift(y, axes=self.dirs[self.ifftshift_before])
        return y.ravel()


class _FFT2D_scipy(_BaseFFTND):
    """Two dimensional Fast-Fourier Transform using scipy"""

    def __init__(
        self,
        dims,
        dirs=(0, 1),
        nffts=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        super().__init__(
            dims=dims,
            dirs=dirs,
            nffts=nffts,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )

        # checks
        if self.ndim < 2:
            raise ValueError("FFT2D requires at least two input dimensions")
        if self.ndirs != 2:
            raise ValueError("FFT2D must be applied along exactly two dimensions")

        self.f1, self.f2 = self.fs
        del self.fs

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before.any():
            x = scipy.fft.ifftshift(x, axes=self.dirs[self.ifftshift_before])
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = scipy.fft.rfft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dirs[-1])
            y[..., 1 : 1 + (self.nffts[-1] - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dirs[-1], -1)
        else:
            y = scipy.fft.fft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        if self.fftshift_after.any():
            y = scipy.fft.fftshift(y, axes=self.dirs[self.fftshift_after])
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after.any():
            x = scipy.fft.ifftshift(x, axes=self.dirs[self.fftshift_after])
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dirs[-1])
            x[..., 1 : 1 + (self.nffts[-1] - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dirs[-1], -1)
            y = scipy.fft.irfft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        else:
            y = scipy.fft.ifft2(x, s=self.nffts, axes=self.dirs, norm="ortho")
        y = np.take(y, range(self.dims[self.dirs[0]]), axis=self.dirs[0])
        y = np.take(y, range(self.dims[self.dirs[1]]), axis=self.dirs[1])
        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before.any():
            y = scipy.fft.fftshift(y, axes=self.dirs[self.ifftshift_before])
        return y.ravel()


def FFT2D(
    dims,
    dirs=(0, 1),
    nffts=None,
    sampling=1.0,
    real=False,
    ifftshift_before=False,
    fftshift_after=False,
    dtype="complex128",
    engine="numpy",
):
    r"""Two dimensional Fast-Fourier Transform.

    Apply two dimensional Fast-Fourier Transform (FFT) to any pair of axes of a
    multi-dimensional array depending on the choice of ``dirs``.

    Using the default NumPy engine, the FFT operator is an overload to either the NumPy
    :py:func:`numpy.fft.fft2` (or :py:func:`numpy.fft.rfft2` for real models) in
    forward mode, and to :py:func:`numpy.fft.ifft2` (or :py:func:`numpy.fft.irfft2`
    for real models) in adjoint mode, or their CuPy equivalents.
    Alternatively, when the SciPy engine is chosen, the overloads are of
    :py:func:`scipy.fft.fft2` (or :py:func:`scipy.fft.rfft2` for real models) in
    forward mode, and to :py:func:`scipy.fft.ifft2` (or :py:func:`scipy.fft.irfft2`
    for real models) in adjoint mode.

    In all cases, the "ortho" scaling (see :py:func:`numpy.fft.fft2`) is used to
    to guarantee that the operator passes the dot-test. When using `real=True`, the
    result of the forward is also multiplied by sqrt(2) for all frequency bins
    except zero and Nyquist along the second direction of ``dirs``, and the input of
    the adjoint is divided by sqrt(2) for the same frequencies.
    If a user is interested in using the unscaled forward FFT, they must pre-multiply
    the operator by an appropriate correction factor.

    For a real valued input signal, it is advised to use the flag ``real=True``
    as it stores the values of the Fourier transform of the last direction at positive
    frequencies only as values at negative frequencies are simply their complex conjugates.


    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dirs : :obj:`tuple`, optional
        Pair of directions along which FFT2D is applied
    nffts : :obj:`tuple` or :obj:`int`, optional
        Number of samples in Fourier Transform for each direction. In case only one
        dimension needs to be specified, use ``None`` for the other dimension in the
        tuple. The direction with None will use ``dims[dir]`` as ``nfft``. When supplying a
        tuple, the order must agree with that of ``dirs``. When a single value is
        passed, it will be used for both directions. As such the default is
        equivalent to ``nffts=(None, None)``.
    sampling : :obj:`tuple` or :obj:`float`, optional
        Sampling steps for each direction. When supplied a single value, it is used
        for both directions. Unlike ``nffts``, ``None``s will not be converted to the
        default value.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real. Note that the real FFT is applied only to the first
        dimension to which the FFT2D operator is applied (last element of
        ``dirs``)
    ifftshift_before : :obj:`tuple` or :obj:`bool`, optional
        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform.
        When passing a single value, the shift will the same for every direction. Pass
        a tuple to specify which dimensions are shifted.
    fftshift_after : :obj:`tuple` or :obj:`bool`, optional
        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero.
        When passing a single value, the shift will the same for every direction. Pass
        a tuple to specify which dimensions are shifted.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``scipy``).
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the ``dtype`` of the operator
        is the corresponding complex type even when a real type is provided.
        In addition, note that the NumPy backend does not support returning ``dtype``s
        different than ``complex128``. As such, when using the NumPy backend, arrays will
        be force-casted to types corresponding to the supplied ``dtype``.
        The SciPy backend supports all precisions natively.
        Under both backends, when a real ``dtype`` is supplied, a real result will be
        enforced on the result of the ``rmatvec`` and the input of the ``matvec``.

    Attributes
    ----------
    dims_fft : :obj:`tuple`
        Shape of the array after the forward, but before linearization. E.g.
        ``y_reshaped = (Op * x.ravel()).reshape(Op.dims_fft)``.
    f1 : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies along ``dir[0]``
    f2 : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies along ``dir[1]``
    real : :obj:`bool`
        When True, uses ``rfft2``/``irfft2``
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
        If ``dims`` has less than two elements.
        If ``dirs`` does not have exactly two elements.
        If ``nffts`` or ``sampling`` are not either a single value or a tuple with
        two elements.
    NotImplementedError
        If ``engine`` is neither ``numpy``, nor ``scipy``.

    Notes
    -----
    The FFT2D operator applies the two-dimensional forward Fourier transform
    to a signal :math:`d(y,x)` in forward mode:

    .. math::
        D(k_y, k_x) = \mathscr{F} (d) = \int \int d(y,x) e^{-j2\pi k_yy}
        e^{-j2\pi k_xx} dy dx

    Similarly, the  two-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_y, k_x)` in adjoint mode:

    .. math::
        d(y,x) = \mathscr{F}^{-1} (D) = \int \int D(k_y, k_x) e^{j2\pi k_yy}
        e^{j2\pi k_xx} dk_y  dk_x

    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform.

    """
    if engine == "numpy":
        f = _FFT2D_numpy(
            dims=dims,
            dirs=dirs,
            nffts=nffts,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    elif engine == "scipy":
        f = _FFT2D_scipy(
            dims=dims,
            dirs=dirs,
            nffts=nffts,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
    else:
        raise NotImplementedError("engine must be numpy or scipy")
    return f
