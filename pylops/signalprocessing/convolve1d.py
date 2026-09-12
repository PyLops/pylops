__all__ = ["Convolve1D"]

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_convolve,
    get_fftconvolve,
    get_oaconvolve,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


def _choose_convfunc(
    x: NDArray,
    method: Literal["direct", "fft", "overlapadd"] | None,
    dims: InputDimsLike,
    axis: int = -1,
) -> tuple[Callable, str]:
    """Choose convolution function

    Choose and return the function handle to be used for convolution
    """
    if len(dims) == 1:
        if method is None:
            method = "direct"
        if method not in ("direct", "fft"):
            msg = "`method` must be direct or fft"
            raise ValueError(msg)
        convfunc = partial(get_convolve(x), method=method)
    else:
        if method is None:
            method = "fft"
        if method == "fft":
            convfunc = partial(get_fftconvolve(x), axes=axis)
        elif method == "overlapadd":
            convfunc = partial(get_oaconvolve(x), axes=axis)
        else:
            msg = "`method` must be fft or overlapadd"
            raise ValueError(msg)
    return convfunc, method


def _pad_along_axis(array: NDArray, pad_size: tuple, axis: int = 0) -> NDArray:
    """Pad an array along a single axis

    Add ``pad_size[0]`` zeros before and ``pad_size[1]`` zeros after ``array``
    along ``axis``, leaving all other axes untouched. Used to shift the filter
    so that its centre lies at the requested ``offset``.
    """
    ncp = get_array_module(array)
    npad = [(0, 0)] * array.ndim
    npad[axis] = pad_size
    return ncp.pad(array, pad_width=npad)


def _broadcast_dimsd(
    dims: InputDimsLike, hshape: tuple, axis: int, nsize: int
) -> tuple:
    """Shape of the data given the shapes of the model and the filter

    The filter is allowed to have more elements than the model along the
    dimensions other than ``axis`` (for example a single wavelet convolved with
    a set of traces): the forward then broadcasts the model over those
    dimensions, and the adjoint sums over them. ``nsize`` is the size of the
    data along ``axis``.
    """
    mshape = list(dims)
    mshape[axis] = 1
    hshape = [1] * len(dims) if len(hshape) == 1 else list(hshape)
    hshape[axis] = 1
    dimsd = list(np.broadcast_shapes(tuple(mshape), tuple(hshape)))
    dimsd[axis] = nsize
    return tuple(dimsd)


def _broadcast_axes(dims: InputDimsLike, dimsd: InputDimsLike, axis: int) -> tuple:
    """Dimensions along which the filter broadcasts the model

    The forward expands the model over these dimensions, so the adjoint has to
    sum over them. They only depend on the shapes of model and data, and are
    therefore evaluated once at construction. ``axis`` is never one of them,
    as the adjoint always crops back to the size of the model along it.
    """
    ax = axis % len(dims)
    return tuple(i for i, d in enumerate(dims) if i != ax and d == 1 and dimsd[i] != 1)


def _take_centered(array: NDArray, size: int, axis: int) -> NDArray:
    """Extract the centre of ``array`` along ``axis``

    Follow the same convention as :py:func:`scipy.signal.fftconvolve` with
    ``mode="same"``, which cannot be used directly when the filter is broadcast
    over the other dimensions of the model (it would also crop those dimensions
    down to the filter's size). A basic slice is used instead of
    :func:`numpy.take` as this routine sits in the matvec of every convolution.
    """
    start = (array.shape[axis] - size) // 2
    indices = [slice(None)] * array.ndim
    indices[axis] = slice(start, start + size)
    return array[tuple(indices)]


class _Convolve1Dshort(LinearOperator):
    r"""1D convolution operator with compact filter (shorter than input signal)"""

    def __init__(
        self,
        dims: int | InputDimsLike,
        h: NDArray,
        offset: int = 0,
        axis: int = -1,
        method: Literal["direct", "fft", "overlapadd"] | None = None,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        ncp = get_array_module(h)
        dims = _value_or_sized_to_tuple(dims)
        dimsd = _broadcast_dimsd(dims, h.shape, axis, dims[axis])
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)
        # ``mode="same"`` crops to the shape of the first input: usable in
        # forward mode only when the filter does not broadcast the model
        self.samemode = tuple(dimsd) == tuple(dims)
        self.sumaxes = _broadcast_axes(dims, dimsd, axis)
        self.axis = axis
        self.nh = h.size if h.ndim == 1 else h.shape[axis]
        if offset > self.nh - 1:
            msg = "`offset` must be smaller than h.shape[axis] - 1"
            raise ValueError(msg)
        self.h = h
        # axis of the filter along which convolution is applied (a 1d filter is
        # broadcast over the other dimensions of the model, so it is always its
        # last - and only - axis)
        haxis = -1 if h.ndim == 1 else axis
        self.offset = 2 * (self.nh // 2 - int(offset))
        if self.nh % 2 == 0:
            self.offset -= 1
        if self.offset != 0:
            self.h = _pad_along_axis(
                self.h,
                (max(self.offset, 0), -min(self.offset, 0)),
                axis=haxis,
            )
        self.hstar = ncp.flip(self.h, axis=haxis)

        # add dimensions to filter to match dimensions of model and data
        if self.h.ndim == 1:
            hdims = np.ones(len(self.dims), dtype=int)
            hdims[self.axis] = len(self.h)
            self.h = self.h.reshape(hdims)
            self.hstar = self.hstar.reshape(hdims)

        # choose method and function handle
        self.convfunc, self.method = _choose_convfunc(h, method, self.dims, self.axis)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if type(self.h) is not type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convfunc, self.method = _choose_convfunc(
                self.h, self.method, self.dims, self.axis
            )
        if self.samemode:
            return self.convfunc(x, self.h, mode="same")
        return _take_centered(self.convfunc(x, self.h), self.dims[self.axis], self.axis)

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if type(self.hstar) is not type(x):
            self.hstar = to_cupy_conditional(x, self.hstar)
            self.convfunc, self.method = _choose_convfunc(
                self.hstar, self.method, self.dims, self.axis
            )
        # the adjoint of a broadcast forward sums over the broadcast dimensions
        y = self.convfunc(x, self.hstar, mode="same")
        return y.sum(axis=self.sumaxes, keepdims=True) if self.sumaxes else y


class _Convolve1Dlong(LinearOperator):
    """1D convolution operator with extended filter (larger than input signal)"""

    def __init__(
        self,
        dims: int | InputDimsLike,
        h: NDArray,
        offset: int = 0,
        axis: int = -1,
        method: Literal["direct", "fft", "overlapadd"] | None = None,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        ncp = get_array_module(h)
        dims = _value_or_sized_to_tuple(dims)
        nh = h.size if h.ndim == 1 else h.shape[axis]
        # the filter is longer than the model along ``axis``, so the data has the
        # same shape as the model with the size of ``axis`` set to the filter length
        dimsd = _broadcast_dimsd(dims, h.shape, axis, nh)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)
        self.sumaxes = _broadcast_axes(dims, dimsd, axis)

        # create filter
        self.axis = axis
        if offset > self.dims[self.axis] - 1:
            msg = "`offset` must be smaller than dims[axis] - 1"
            raise ValueError(msg)
        self.nh = nh
        self.h = h
        # axis of the filter along which convolution is applied (see _Convolve1Dshort)
        haxis = -1 if h.ndim == 1 else axis
        self.offset = 2 * (self.dims[self.axis] // 2 - int(offset))
        if self.dims[self.axis] % 2 == 0:
            self.offset -= 1
        self.hstar = ncp.flip(self.h, axis=haxis)

        self.pad = np.zeros((len(dims), 2), dtype=int)
        self.pad[self.axis, 0] = max(self.offset, 0)
        self.pad[self.axis, 1] = -min(self.offset, 0)

        self.padd = np.zeros((len(dims), 2), dtype=int)
        self.padd[self.axis, 1] = max(self.offset, 0)
        self.padd[self.axis, 0] = -min(self.offset, 0)

        # add dimensions to filter to match dimensions of model and data
        if self.h.ndim == 1:
            hdims = np.ones(len(self.dims), dtype=int)
            hdims[self.axis] = len(self.h)
            self.h = self.h.reshape(hdims)
            self.hstar = self.hstar.reshape(hdims)

        # ``mode="same"`` crops to the shape of the first input, here the filter:
        # usable in the forward only when the filter already has the data's shape
        self.samemode = tuple(self.h.shape) == tuple(self.dimsd)

        # choose method and function handle
        self.convfunc, self.method = _choose_convfunc(h, method, self.dims, self.axis)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if type(self.h) is not type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convfunc, self.method = _choose_convfunc(
                self.h, self.method, self.dims, self.axis
            )
        x = ncp.pad(x, self.pad)
        if self.samemode:
            return self.convfunc(self.h, x, mode="same")
        return _take_centered(self.convfunc(self.h, x), self.nh, self.axis)

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if type(self.hstar) is not type(x):
            self.hstar = to_cupy_conditional(x, self.hstar)
            self.convfunc, self.method = _choose_convfunc(
                self.hstar, self.method, self.dims, self.axis
            )
        x = ncp.pad(x, self.padd)
        y = self.convfunc(self.hstar, x)
        # the adjoint of a broadcast forward sums over the broadcast dimensions
        y = _take_centered(y, self.dims[self.axis], self.axis)
        return y.sum(axis=self.sumaxes, keepdims=True) if self.sumaxes else y


class Convolve1D(LinearOperator):
    r"""1D convolution operator.

    Apply one-dimensional convolution with i) a compact filter (shorter than input signal) or
    ii) an extended filter (larger than input signal) to model (and data) along an ``axis``
    of a multi-dimensional array.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension of the model
    h : :obj:`numpy.ndarray`
        Filter to be convolved to input signal. Either a 1d array, which is
        applied to every 1d slice of the model taken along ``axis``, or an array
        with the same number of dimensions as the model, which allows using a
        different filter for each slice. In the latter case the filter may also
        be larger than the model along the dimensions other than ``axis``,
        provided the model has size 1 along those dimensions: the forward then
        broadcasts the model over them and the adjoint sums over them.
    offset : :obj:`int`
        Index of the center of the filter
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which convolution is applied
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct``, ``fft``,
        or ``overlapadd``). Note that only ``direct`` and ``fft`` are allowed
        for a one-dimensional model, whilst ``fft`` and ``overlapadd`` are
        allowed for a multi-dimensional model. If ``None``, the method is
        chosen automatically (``direct`` for 1-dimensional inputs and ``fft``
        for N-dimensional inputs)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    nh : :obj:`int`
        Length of the filter
    hstar : :obj:`numpy.ndarray`
        Time-reversed filter used in adjoint
    convfunc : :obj:`callable`
        Function handler used to perform convolution
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. Obtained by
        broadcasting ``dims`` against the shape of ``h`` over the dimensions
        other than ``axis``, whose size is that of the model for a compact
        filter and that of the filter for an extended one. Same as ``dims`` for
        a 1d compact filter.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ValueError
        If ``offset`` is bigger than the size along ``axis`` of the filter
        (compact filter) or of the model (extended filter), minus one
    ValueError
        If ``method`` provided is not allowed
    ValueError
        If the shape of ``h`` cannot be broadcast against ``dims`` over the
        dimensions other than ``axis``

    Notes
    -----
    The Convolve1D operator applies convolution between the input signal
    :math:`x(t)` and a filter kernel :math:`h(t)` in forward model:

    .. math::
        y(t) = \int\limits_{-\infty}^{\infty} h(t-\tau) x(\tau) \,\mathrm{d}\tau

    This operation can be discretized as follows

    .. math::
        y[n] = \sum_{m=-\infty}^{\infty} h[n-m] x[m]

    as well as performed in the frequency domain.

    .. math::
        Y(f) = \mathscr{F} (h(t)) * \mathscr{F} (x(t))

    For one dimensional inputs, Convolve1D operator uses
    :py:func:`scipy.signal.convolve`, which automatically chooses the best
    domain for the operation to be carried out. For signals in 2 or more
    dimensions, the default is instead the fft implementation
    :py:func:`scipy.signal.fftconvolve`, as this routine efficently operates on
    multi-dimensional arrays; the overlap-add implementation
    :py:func:`scipy.signal.oaconvolve` can be selected via the ``method``
    parameter, and may be faster when the filter is much shorter than the
    signal.

    As the adjoint of convolution is correlation, Convolve1D operator applies
    correlation in the adjoint mode.

    In time domain:

    .. math::
        x(t) = \int\limits_{-\infty}^{\infty} h(t+\tau) x(\tau) \,\mathrm{d}\tau

    or in frequency domain:

    .. math::
        y(t) = \mathscr{F}^{-1} (H(f)^* * X(f))

    """

    def __init__(
        self,
        dims: int | InputDimsLike,
        h: NDArray,
        offset: int = 0,
        axis: int = -1,
        method: Literal["direct", "fft", "overlapadd"] | None = None,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        nh = h.size if h.ndim == 1 else h.shape[axis]
        if nh <= _value_or_sized_to_tuple(dims)[axis]:
            convop = _Convolve1Dshort
        else:
            convop = _Convolve1Dlong
        Op = convop(
            dims=dims,
            h=h,
            offset=offset,
            axis=axis,
            method=method,
            dtype=dtype,
        )
        super().__init__(Op=Op, name=name)

    def _matvec(self, x: NDArray) -> NDArray:
        return super()._matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return super()._rmatvec(x)
