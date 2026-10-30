__all__ = ["ConvolveND"]

from typing import Literal

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_convolve,
    get_correlate,
    get_normalize_axis_index,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class ConvolveND(LinearOperator):
    r"""ND convolution operator.

    Apply n-dimensional convolution with a compact filter to model
    (and data) along the ``axes`` of a n-dimensional array.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    h : :obj:`numpy.ndarray`
        nd compact filter to be convolved to input signal
    offset : :obj:`tuple`, optional
        Indices of the center of the compact filter
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which convolution is applied. If ``None``, use the last
        ``h.ndim`` axes.
    method : :obj:`str`, optional
        Method used to calculate the convolution (``auto``, ``direct`` or ``fft``)
        - see :func:`scipy.signal.convolve` for details.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    nh : :obj:`tuple`
        Length of the filter
    convolve : :obj:`callable`
        Convolution function
    correlate : :obj:`callable`
        Correlation function
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ValueError
        If ``h`` has more dimensions than ``dims``, or ``axes`` does not
        contain one unique model axis per filter dimension.

    Notes
    -----
    The ConvolveND operator applies n-dimensional convolution
    between the input signal :math:`d(x_1, x_2, ..., x_N)` and a compact
    filter kernel :math:`h(x_1, x_2, ..., x_N)` in forward model. This is a
    straighforward extension to multiple dimensions of
    :obj:`pylops.signalprocessing.Convolve2D` operator.

    """

    def __init__(
        self,
        dims: int | InputDimsLike,
        h: NDArray,
        offset: InputDimsLike | None = None,
        axes: InputDimsLike | None = None,
        method: Literal["auto", "direct", "fft"] | None = "fft",
        dtype: DTypeLike = "float64",
        name: str = "C",
    ):
        dims = _value_or_sized_to_tuple(dims)
        if h.ndim > len(dims):
            msg = "h cannot have more dimensions than dims"
            raise ValueError(msg)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axes = (
            np.arange(len(self.dims) - h.ndim, len(self.dims))
            if axes is None
            else np.array(
                [get_normalize_axis_index()(ax, len(self.dims)) for ax in axes]
            )
        )
        if len(self.axes) != h.ndim:
            msg = "axes must contain one axis per dimension of h"
            raise ValueError(msg)
        if len(set(self.axes)) != len(self.axes):
            msg = "axes must be unique"
            raise ValueError(msg)
        self.h = h
        hshape = np.array(self.h.shape)

        # padding
        if offset is None:
            offsetpad = np.zeros(self.h.ndim, dtype=int)
        else:
            offsetpad = np.asarray(offset, dtype=int)
        self.offset = 2 * (hshape // 2 - offsetpad)
        pad = [(0, 0) for _ in range(self.h.ndim)]
        dopad = False
        for inh, nh in enumerate(hshape):
            if nh % 2 == 0:
                self.offset[inh] -= 1
            if self.offset[inh] != 0:
                pad[inh] = (
                    self.offset[inh] if self.offset[inh] > 0 else 0,
                    -self.offset[inh] if self.offset[inh] < 0 else 0,
                )
                dopad = True
        if dopad:
            ncp = get_array_module(h)
            self.h = ncp.pad(self.h, pad, mode="constant")
        self.nh = self.h.shape

        # find out which directions are used for convolution and define offsets
        self.h = self.h.transpose(tuple(np.argsort(self.axes)))
        if len(self.dims) != len(self.nh):
            dimsh = np.ones(len(self.dims), dtype=int)
            for iax, ax in enumerate(self.axes):
                dimsh[ax] = self.nh[iax]
            self.h = self.h.reshape(dimsh)

        # convolve and correlate functions
        self.convolve = get_convolve(h)
        self.correlate = get_correlate(h)
        self.method = method

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) is not type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        return self.convolve(x, self.h, mode="same", method=self.method)

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) is not type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        return self.correlate(x, self.h, mode="same", method=self.method)
