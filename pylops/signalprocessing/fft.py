__all__ = ["FFT"]

import logging
import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.fft

from pylops import LinearOperator
from pylops.signalprocessing._baseffts import _BaseFFT, _FFTNorms
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

pyfftw_message = deps.pyfftw_import("the fft module")

if pyfftw_message is None:
    import pyfftw

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class _FFT_numpy(_BaseFFT):
    """One dimensional Fast-Fourier Transform using NumPy"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        nfft: Optional[int] = None,
        sampling: float = 1.0,
        norm: str = "ortho",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
    ) -> None:
        super().__init__(
            dims=dims,
            axis=axis,
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

        self._norm_kwargs = {"norm": None}  # equivalent to "backward" in Numpy/Scipy
        if self.norm is _FFTNorms.ORTHO:
            self._norm_kwargs["norm"] = "ortho"
        elif self.norm is _FFTNorms.NONE:
            self._scale = self.nfft
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / self.nfft

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if self.ifftshift_before:
            x = np.fft.ifftshift(x, axes=self.axis)
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = np.fft.rfft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.axis)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.axis, -1)
        else:
            y = np.fft.fft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        if self.norm is _FFTNorms.ONE_OVER_N:
            y *= self._scale
        if self.fftshift_after:
            y = np.fft.fftshift(y, axes=self.axis)
        y = y.astype(self.cdtype)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.fftshift_after:
            x = np.fft.ifftshift(x, axes=self.axis)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.axis)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.axis, -1)
            y = np.fft.irfft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        else:
            y = np.fft.ifft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        if self.norm is _FFTNorms.NONE:
            y *= self._scale

        if self.nfft > self.dims[self.axis]:
            y = np.take(y, range(0, self.dims[self.axis]), axis=self.axis)
        elif self.nfft < self.dims[self.axis]:
            y = np.pad(y, self.ifftpad)

        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before:
            y = np.fft.fftshift(y, axes=self.axis)
        y = y.astype(self.rdtype)
        return y

    def __truediv__(self, y: npt.ArrayLike) -> npt.ArrayLike:
        if self.norm is not _FFTNorms.ORTHO:
            return self._rmatvec(y) / self._scale
        return self._rmatvec(y)


class _FFT_scipy(_BaseFFT):
    """One dimensional Fast-Fourier Transform using SciPy"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        nfft: Optional[int] = None,
        sampling: float = 1.0,
        norm: str = "ortho",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
    ) -> None:
        super().__init__(
            dims=dims,
            axis=axis,
            nfft=nfft,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )

        self._norm_kwargs = {"norm": None}  # equivalent to "backward" in Numpy/Scipy
        if self.norm is _FFTNorms.ORTHO:
            self._norm_kwargs["norm"] = "ortho"
        elif self.norm is _FFTNorms.NONE:
            self._scale = self.nfft
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / self.nfft

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if self.ifftshift_before:
            x = scipy.fft.ifftshift(x, axes=self.axis)
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = scipy.fft.rfft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.axis)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.axis, -1)
        else:
            y = scipy.fft.fft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        if self.norm is _FFTNorms.ONE_OVER_N:
            y *= self._scale
        if self.fftshift_after:
            y = scipy.fft.fftshift(y, axes=self.axis)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.fftshift_after:
            x = scipy.fft.ifftshift(x, axes=self.axis)
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.axis)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.axis, -1)
            y = scipy.fft.irfft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        else:
            y = scipy.fft.ifft(x, n=self.nfft, axis=self.axis, **self._norm_kwargs)
        if self.norm is _FFTNorms.NONE:
            y *= self._scale

        if self.nfft > self.dims[self.axis]:
            y = np.take(y, range(0, self.dims[self.axis]), axis=self.axis)
        elif self.nfft < self.dims[self.axis]:
            y = np.pad(y, self.ifftpad)

        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before:
            y = scipy.fft.fftshift(y, axes=self.axis)
        return y

    def __truediv__(self, y):
        if self.norm is not _FFTNorms.ORTHO:
            return self._rmatvec(y) / self._scale
        return self._rmatvec(y)


class _FFT_fftw(_BaseFFT):
    """One dimensional Fast-Fourier Transform using pyFFTW"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        nfft: Optional[int] = None,
        sampling: float = 1.0,
        norm: str = "ortho",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
        **kwargs_fftw,
    ) -> None:
        if np.dtype(dtype) == np.float16:
            warnings.warn(
                "fftw backend is unavailable with float16 dtype. Will use float32."
            )
            dtype = np.float32

        for badop in ["ortho", "normalise_idft"]:
            if badop in kwargs_fftw:
                if badop == "ortho" and norm == "ortho":
                    continue
                warnings.warn(
                    f"FFTW option '{badop}' will be overwritten by norm={norm}"
                )
                del kwargs_fftw[badop]

        super().__init__(
            dims=dims,
            axis=axis,
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
                f"fftw backend returns complex128 dtype. To respect the passed dtype, data will be cast to {self.cdtype}."
            )

        dims_t = list(self.dims)
        dims_t[self.axis] = self.nfft
        self.dims_t = dims_t

        # define padding(fftw requires the user to provide padded input signal)
        self.pad = np.zeros((self.ndim, 2), dtype=int)
        if self.real:
            if self.nfft % 2:
                self.pad[self.axis, 1] = (
                    2 * (self.dimsd[self.axis] - 1) + 1 - self.dims[self.axis]
                )
            else:
                self.pad[self.axis, 1] = (
                    2 * (self.dimsd[self.axis] - 1) - self.dims[self.axis]
                )
        else:
            self.pad[self.axis, 1] = self.dimsd[self.axis] - self.dims[self.axis]
        self.dopad = True if np.sum(self.pad) > 0 else False

        # create empty arrays and plans for fft/ifft
        self.x = pyfftw.empty_aligned(
            self.dims_t, dtype=self.rdtype if real else self.cdtype
        )
        self.y = pyfftw.empty_aligned(self.dimsd, dtype=self.cdtype)

        # Use FFTW without norm-related keywords above. In this case, FFTW standard
        # behavior is to scale with 1/N on the inverse transform. The _scale below
        # converts the default bevavior to that of ``norm``. Keywords ``ortho`` and
        # ``normalise_idft`` do not seem to be available over all versions of pyFFTW
        # so best to avoid them.
        if self.norm is _FFTNorms.ORTHO:
            self._scale = np.sqrt(1.0 / self.nfft)
        elif self.norm is _FFTNorms.NONE:
            self._scale = self.nfft
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / self.nfft

        self.fftplan = pyfftw.FFTW(
            self.x, self.y, axes=(self.axis,), direction="FFTW_FORWARD", **kwargs_fftw
        )
        self.ifftplan = pyfftw.FFTW(
            self.y, self.x, axes=(self.axis,), direction="FFTW_BACKWARD", **kwargs_fftw
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if self.ifftshift_before:
            x = np.fft.ifftshift(x, axes=self.axis)
        if not self.clinear:
            x = np.real(x)
        if self.dopad:
            x = np.pad(x, self.pad, "constant", constant_values=0)
        elif self.doifftpad:
            x = np.take(x, range(0, self.nfft), axis=self.axis)

        # self.fftplan() always uses byte-alligned self.x as input array and
        # returns self.y as output array. As such, self.x must be copied so as
        # not to be overwritten on a subsequent call to _matvec.
        np.copyto(self.x, x)
        y = self.fftplan().copy()
        if self.norm is not _FFTNorms.NONE:
            y *= self._scale

        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.axis)
            y[..., 1 : 1 + (self.nfft - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.axis, -1)
        if self.fftshift_after:
            y = np.fft.fftshift(y, axes=self.axis)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.fftshift_after:
            x = np.fft.ifftshift(x, axes=self.axis)

        # self.ifftplan() always uses byte-alligned self.y as input array.
        # We copy here so we don't need to copy again in the case of `real=True`,
        # which only performs operations that preserve byte-allignment.
        np.copyto(self.y, x)
        x = self.y  # Update reference only

        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = np.swapaxes(x, -1, self.axis)
            x[..., 1 : 1 + (self.nfft - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.axis, -1)

        # self.ifftplan() always returns self.x, which must be copied so as not
        # to be overwritten on a subsequent call to _rmatvec.
        y = self.ifftplan().copy()
        if self.norm is _FFTNorms.ORTHO:
            y /= self._scale
        elif self.norm is _FFTNorms.NONE:
            y *= self._scale

        if self.nfft > self.dims[self.axis]:
            y = np.take(y, range(0, self.dims[self.axis]), axis=self.axis)
        elif self.nfft < self.dims[self.axis]:
            y = np.pad(y, self.ifftpad)

        if self.ifftshift_before:
            y = np.fft.fftshift(y, axes=self.axis)
        if not self.clinear:
            y = np.real(y)
        return y

    def __truediv__(self, y: npt.ArrayLike) -> npt.ArrayLike:
        if self.norm is _FFTNorms.ORTHO:
            return self._rmatvec(y)
        return self._rmatvec(y) / self._scale


def FFT(
    dims: Union[int, InputDimsLike],
    axis: int = -1,
    nfft: Optional[int] = None,
    sampling: float = 1.0,
    norm: str = "ortho",
    real: bool = False,
    ifftshift_before: bool = False,
    fftshift_after: bool = False,
    engine: str = "numpy",
    dtype: DTypeLike = "complex128",
    name: str = "F",
    **kwargs_fftw,
) -> LinearOperator:
    r"""One dimensional Fast-Fourier Transform.

    Apply Fast-Fourier Transform (FFT) along an ``axis`` of a
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

    When using ``real=True``, the result of the forward is also multiplied by
    :math:`\sqrt{2}` for all frequency bins except zero and Nyquist, and the input of
    the adjoint is multiplied by :math:`1 / \sqrt{2}` for the same frequencies.

    For a real valued input signal, it is advised to use the flag ``real=True``
    as it stores the values of the Fourier transform at positive frequencies only as
    values at negative frequencies are simply their complex conjugates.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which FFT is applied
    nfft : :obj:`int`, optional
        Number of samples in Fourier Transform (same as input if ``nfft=None``)
    sampling : :obj:`float`, optional
        Sampling step ``dt``.
    norm : `{"ortho", "none", "1/n"}`, optional
        .. versionadded:: 1.17.0

        - "ortho": Scales forward and adjoint FFT transforms with :math:`1/\sqrt{N_F}`,
          where :math:`N_F` is the number of samples in the Fourier domain given by ``nfft``.

        - "none": Does not scale the forward or the adjoint FFT transforms.

        - "1/n": Scales both the forward and adjoint FFT transforms by
          :math:`1/N_F`.

        .. note:: For "none" and "1/n", the operator is not unitary, that is, the
          adjoint is not the inverse. To invert the operator, simply use ``Op \ y``.

    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    ifftshift_before : :obj:`bool`, optional
        .. versionadded:: 1.17.0

        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform.
        Defaults to not applying ifftshift.
    fftshift_after : :obj:`bool`, optional
        .. versionadded:: 1.17.0

        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``fftw``, or ``scipy``). Choose
        ``numpy`` when working with cupy arrays.

        .. note:: Since version 1.17.0, accepts "scipy".

    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the ``dtype`` of the operator
        is the corresponding complex type even when a real type is provided.
        In addition, note that neither the NumPy nor the FFTW backends supports
        returning ``dtype`` different than ``complex128``. As such, when using either
        backend, arrays will be force-casted to types corresponding to the supplied ``dtype``.
        The SciPy backend supports all precisions natively.
        Under all backends, when a real ``dtype`` is supplied, a real result will be
        enforced on the result of the ``rmatvec`` and the input of the ``matvec``.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)
    **kwargs_fftw
            Arbitrary keyword arguments
            for :py:class:`pyfftw.FTTW`

    Attributes
    ----------
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before linearization.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    f : :obj:`numpy.ndarray`
        Discrete Fourier Transform sample frequencies
    real : :obj:`bool`
        When ``True``, uses ``rfft``/``irfft``
    rdtype : :obj:`bool`
        Expected input type to the forward
    cdtype : :obj:`bool`
        Output type of the forward. Complex equivalent to ``rdtype``.
    shape : :obj:`tuple`
        Operator shape
    clinear : :obj:`bool`
        .. versionadded:: 1.17.0

        Operator is complex-linear. Is false when either ``real=True`` or when
        ``dtype`` is not a complex type.
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    ValueError
        - If ``dims`` is provided and ``axis`` is bigger than ``len(dims)``.
        - If ``norm`` is not one of "ortho", "none", or "1/n".
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``fftw``, nor ``scipy``.

    See Also
    --------
    FFT2D: Two-dimensional FFT
    FFTND: N-dimensional FFT


    Notes
    -----
    The FFT operator (using ``norm="ortho"``) applies the forward Fourier transform to
    a signal
    :math:`d(t)` in forward mode:

    .. math::
        D(f) = \mathscr{F} (d) = \frac{1}{\sqrt{N_F}} \int\limits_{-\infty}^\infty d(t) e^{-j2\pi ft} \,\mathrm{d}t

    Similarly, the inverse Fourier transform is applied to the Fourier spectrum
    :math:`D(f)` in adjoint mode:

    .. math::
        d(t) = \mathscr{F}^{-1} (D) = \frac{1}{\sqrt{N_F}} \int\limits_{-\infty}^\infty D(f) e^{j2\pi ft} \,\mathrm{d}f

    where :math:`N_F` is the number of samples in the Fourier domain ``nfft``.
    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform. Note that the FFT operator
    (using ``norm="ortho"``) is a special operator in that the adjoint is also
    the inverse of the forward mode. For other norms, this does not hold (see ``norm``
    help). However, for any norm, the Fourier transform is Hermitian for real input
    signals.

    """
    if engine == "fftw" and pyfftw_message is None:
        f = _FFT_fftw(
            dims,
            axis=axis,
            nfft=nfft,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            **kwargs_fftw,
        )
    elif engine == "numpy" or (engine == "fftw" and pyfftw_message is not None):
        if engine == "fftw" and pyfftw_message is not None:
            logging.warning(pyfftw_message)
        f = _FFT_numpy(
            dims,
            axis=axis,
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
            axis=axis,
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
    f.name = name
    return f
