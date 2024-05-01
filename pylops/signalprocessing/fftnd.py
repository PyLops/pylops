__all__ = ["FFTND"]

import logging
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from pylops.signalprocessing._baseffts import _BaseFFTND, _FFTNorms
from pylops.utils.backend import get_sp_fft
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class _FFTND_numpy(_BaseFFTND):
    """N-dimensional Fast-Fourier Transform using NumPy"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axes: Union[int, InputDimsLike] = (-3, -2, -1),
        nffts: Optional[Union[int, InputDimsLike]] = None,
        sampling: Union[float, Sequence[float]] = 1.0,
        norm: str = "ortho",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
        **kwargs_fft,
    ) -> None:
        super().__init__(
            dims=dims,
            axes=axes,
            nffts=nffts,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        if self.cdtype != np.complex128:
            warnings.warn(
                f"numpy backend always returns complex128 dtype. To respect the passed dtype, data will be cast to {self.cdtype}."
            )
        self._kwargs_fft = kwargs_fft
        self._norm_kwargs = {"norm": None}  # equivalent to "backward" in Numpy/Scipy
        if self.norm is _FFTNorms.ORTHO:
            self._norm_kwargs["norm"] = "ortho"
        elif self.norm is _FFTNorms.NONE:
            self._scale = np.prod(self.nffts)
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / np.prod(self.nffts)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if self.ifftshift_before.any():
            x = np.fft.ifftshift(x, axes=self.axes[self.ifftshift_before])
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = np.fft.rfftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.axes[-1])
            y[..., 1 : 1 + (self.nffts[-1] - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.axes[-1], -1)
        else:
            y = np.fft.fftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        if self.norm is _FFTNorms.ONE_OVER_N:
            y *= self._scale
        y = y.astype(self.cdtype)
        if self.fftshift_after.any():
            y = np.fft.fftshift(y, axes=self.axes[self.fftshift_after])
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.fftshift_after.any():
            x = np.fft.ifftshift(x, axes=self.axes[self.fftshift_after])
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.axes[-1])
            x[..., 1 : 1 + (self.nffts[-1] - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.axes[-1], -1)
            y = np.fft.irfftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        else:
            y = np.fft.ifftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        if self.norm is _FFTNorms.NONE:
            y *= self._scale
        for ax, nfft in zip(self.axes, self.nffts):
            if nfft > self.dims[ax]:
                y = np.take(y, range(self.dims[ax]), axis=ax)
        if self.doifftpad:
            y = np.pad(y, self.ifftpad)
        if not self.clinear:
            y = np.real(y)
        y = y.astype(self.rdtype)
        if self.ifftshift_before.any():
            y = np.fft.fftshift(y, axes=self.axes[self.ifftshift_before])
        return y

    def __truediv__(self, y: npt.ArrayLike) -> npt.ArrayLike:
        if self.norm is not _FFTNorms.ORTHO:
            return self._rmatvec(y) / self._scale
        return self._rmatvec(y)


class _FFTND_scipy(_BaseFFTND):
    """N-dimensional Fast-Fourier Transform using SciPy"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axes: Union[int, InputDimsLike] = (-3, -2, -1),
        nffts: Optional[Union[int, InputDimsLike]] = None,
        sampling: Union[float, Sequence[float]] = 1.0,
        norm: str = "ortho",
        real: bool = False,
        ifftshift_before: bool = False,
        fftshift_after: bool = False,
        dtype: DTypeLike = "complex128",
        **kwargs_fft,
    ) -> None:
        super().__init__(
            dims=dims,
            axes=axes,
            nffts=nffts,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )
        self._kwargs_fft = kwargs_fft
        self._norm_kwargs = {"norm": None}  # equivalent to "backward" in Numpy/Scipy
        if self.norm is _FFTNorms.ORTHO:
            self._norm_kwargs["norm"] = "ortho"
        elif self.norm is _FFTNorms.NONE:
            self._scale = np.prod(self.nffts)
        elif self.norm is _FFTNorms.ONE_OVER_N:
            self._scale = 1.0 / np.prod(self.nffts)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        sp_fft = get_sp_fft(x)
        if self.ifftshift_before.any():
            x = sp_fft.ifftshift(x, axes=self.axes[self.ifftshift_before])
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = sp_fft.rfftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.axes[-1])
            y[..., 1 : 1 + (self.nffts[-1] - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.axes[-1], -1)
        else:
            y = sp_fft.fftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        if self.norm is _FFTNorms.ONE_OVER_N:
            y *= self._scale
        if self.fftshift_after.any():
            y = sp_fft.fftshift(y, axes=self.axes[self.fftshift_after])
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        sp_fft = get_sp_fft(x)
        if self.fftshift_after.any():
            x = sp_fft.ifftshift(x, axes=self.axes[self.fftshift_after])
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.axes[-1])
            x[..., 1 : 1 + (self.nffts[-1] - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.axes[-1], -1)
            y = sp_fft.irfftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        else:
            y = sp_fft.ifftn(
                x, s=self.nffts, axes=self.axes, **self._norm_kwargs, **self._kwargs_fft
            )
        if self.norm is _FFTNorms.NONE:
            y *= self._scale
        for ax, nfft in zip(self.axes, self.nffts):
            if nfft > self.dims[ax]:
                y = np.take(y, range(self.dims[ax]), axis=ax)
        if self.doifftpad:
            y = np.pad(y, self.ifftpad)
        if not self.clinear:
            y = np.real(y)
        if self.ifftshift_before.any():
            y = sp_fft.fftshift(y, axes=self.axes[self.ifftshift_before])
        return y

    def __truediv__(self, y: npt.ArrayLike) -> npt.ArrayLike:
        if self.norm is not _FFTNorms.ORTHO:
            return self._rmatvec(y) / self._scale
        return self._rmatvec(y)


def FFTND(
    dims: Union[int, InputDimsLike],
    axes: Union[int, InputDimsLike] = (-3, -2, -1),
    nffts: Optional[Union[int, InputDimsLike]] = None,
    sampling: Union[float, Sequence[float]] = 1.0,
    norm: str = "ortho",
    real: bool = False,
    ifftshift_before: bool = False,
    fftshift_after: bool = False,
    engine: str = "scipy",
    dtype: DTypeLike = "complex128",
    name: str = "F",
    **kwargs_fft,
):
    r"""N-dimensional Fast-Fourier Transform.

    Apply N-dimensional Fast-Fourier Transform (FFT) to any n ``axes``
    of a multi-dimensional array.

    Using the default NumPy engine, the FFT operator is an overload to either the NumPy
    :py:func:`numpy.fft.fftn` (or :py:func:`numpy.fft.rfftn` for real models) in
    forward mode, and to :py:func:`numpy.fft.ifftn` (or :py:func:`numpy.fft.irfftn`
    for real models) in adjoint mode, or their CuPy equivalents.
    Alternatively, when the SciPy engine is chosen, the overloads are of
    :py:func:`scipy.fft.fftn` (or :py:func:`scipy.fft.rfftn` for real models) in
    forward mode, and to :py:func:`scipy.fft.ifftn` (or :py:func:`scipy.fft.irfftn`
    for real models) in adjoint mode.

    When using ``real=True``, the result of the forward is also multiplied by
    :math:`\sqrt{2}` for all frequency bins except zero and Nyquist along the last
    ``axes``, and the input of the adjoint is multiplied by
    :math:`1 / \sqrt{2}` for the same frequencies.

    For a real valued input signal, it is advised to use the flag ``real=True``
    as it stores the values of the Fourier transform of the last axis in ``axes`` at positive
    frequencies only as values at negative frequencies are simply their complex conjugates.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes (or axis) along which FFTND is applied
    nffts : :obj:`tuple` or :obj:`int`, optional
        Number of samples in Fourier Transform for each axis in ``axes``. In case only one
        dimension needs to be specified, use ``None`` for the other dimension in the
        tuple. An axis with ``None`` will use ``dims[axis]`` as ``nfft``.
        When supplying a tuple, the length must agree with that
        of ``axes``. When a single value is passed, it will be used for all
        ``axes`. As such the default is equivalent to ``nffts=(None, ..., None)``.
    sampling : :obj:`tuple` or :obj:`float`, optional
        Sampling steps for each direction. When supplied a single value, it is used
        for all directions. Unlike ``nffts``, any ``None`` will not be converted to the
        default value.
    norm : `{"ortho", "none", "1/n"}`, optional
        .. versionadded:: 1.17.0

        - "ortho": Scales forward and adjoint FFT transforms with :math:`1/\sqrt{N_F}`,
          where :math:`N_F` is the number of samples in the Fourier domain given by
          product of all elements of ``nffts``.

        - "none": Does not scale the forward or the adjoint FFT transforms.

        - "1/n": Scales both the forward and adjoint FFT transforms by
          :math:`1/N_F`.

        .. note:: For "none" and "1/n", the operator is not unitary, that is, the
          adjoint is not the inverse. To invert the operator, simply use ``Op \ y``.

    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real. Note that the real FFT is applied only to the first
        dimension to which the FFTND operator is applied (last element of
        ``axes``)
    ifftshift_before : :obj:`tuple` or :obj:`bool`, optional
        .. versionadded:: 1.17.0

        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform.
        When passing a single value, the shift will the same for every direction. Pass
        a tuple to specify which dimensions are shifted.
    fftshift_after : :obj:`tuple` or :obj:`bool`, optional
        .. versionadded:: 1.17.0

        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero.
        When passing a single value, the shift will the same for every direction. Pass
        a tuple to specify which dimensions are shifted.
    engine : :obj:`str`, optional
        .. versionadded:: 1.17.0

        Engine used for fft computation (``numpy`` or ``scipy``).
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the ``dtype`` of the operator
        is the corresponding complex type even when a real type is provided.
        In addition, note that the NumPy backend does not support returning ``dtype``
        different than ``complex128``. As such, when using the NumPy backend, arrays will
        be force-cast to types corresponding to the supplied ``dtype``.
        The SciPy backend supports all precisions natively.
        Under both backends, when a real ``dtype`` is supplied, a real result will be
        enforced on the result of the ``rmatvec`` and the input of the ``matvec``.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)
    **kwargs_fft
            Arbitrary keyword arguments to be passed to the selected fft method

    Attributes
    ----------
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before linearization.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    fs : :obj:`tuple`
        Each element of the tuple corresponds to the Discrete Fourier Transform
        sample frequencies along the respective direction given by ``axes``.
    real : :obj:`bool`
        When ``True``, uses ``rfftn``/``irfftn``
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

    See Also
    --------
    FFT: One-dimensional FFT
    FFT2D: Two-dimensional FFT

    Raises
    ------
    ValueError
        - If ``nffts`` or ``sampling`` are not either a single value or tuple with
          the same dimension ``axes``.
        - If ``norm`` is not one of "ortho", "none", or "1/n".
    NotImplementedError
        If ``engine`` is neither ``numpy``, nor ``scipy``.

    Notes
    -----
    The FFTND operator (using ``norm="ortho"``) applies the N-dimensional forward
    Fourier transform to a multi-dimensional array. Considering an N-dimensional
    signal :math:`d(x_1, \ldots, x_N)`. The FFTND in forward mode is:

    .. math::
        D(k_1, \ldots, k_N) = \mathscr{F} (d) = \frac{1}{\sqrt{N_F}}
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        d(x_1, \ldots, x_N)
        e^{-j2\pi k_1 x_1} \cdots
        e^{-j 2 \pi k_N x_N}  \,\mathrm{d}x_1 \cdots \mathrm{d}x_N

    Similarly, the  three-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_z, k_y, k_x)` in adjoint mode:

    .. math::
        d(x_1, \ldots, x_N) = \mathscr{F}^{-1} (D) = \frac{1}{\sqrt{N_F}}
        \int\limits_{-\infty}^\infty \cdots \int\limits_{-\infty}^\infty
        D(k_1, \ldots, k_N)
        e^{-j2\pi k_1 x_1} \cdots
        e^{-j 2 \pi k_N x_N} \,\mathrm{d}k_1 \cdots  \mathrm{d}k_N

    where :math:`N_F` is the number of samples in the Fourier domain given by the
    product of the element of ``nffts``.
    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform. Note that the FFTND operator
    (using ``norm="ortho"``) is a special operator in that the adjoint is also
    the inverse of the forward mode. For other norms, this does not hold (see ``norm``
    help). However, for any norm, the N-dimensional Fourier transform is Hermitian
    for real input signals.

    """
    if engine == "numpy":
        f = _FFTND_numpy(
            dims=dims,
            axes=axes,
            nffts=nffts,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            **kwargs_fft,
        )
    elif engine == "scipy":
        f = _FFTND_scipy(
            dims=dims,
            axes=axes,
            nffts=nffts,
            sampling=sampling,
            norm=norm,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
            **kwargs_fft,
        )
    else:
        raise NotImplementedError("engine must be numpy or scipy")
    f.name = name
    return f
