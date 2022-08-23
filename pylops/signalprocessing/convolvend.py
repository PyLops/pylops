__all__ = ["ConvolveND"]

from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_convolve,
    get_correlate,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped


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

        Axes along which convolution is applied
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct`` or ``fft``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

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
        dims: Union[int, List],
        h: npt.ArrayLike,
        offset: int = None,
        axes: Tuple = (-2, -1),
        method: str = "fft",
        dtype: str = "float64",
        name: str = "C",
    ):
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axes = (
            np.arange(len(self.dims))
            if axes is None
            else np.array([normalize_axis_index(ax, len(self.dims)) for ax in axes])
        )
        self.h = h
        self.nh = np.array(self.h.shape)

        # padding
        if offset is None:
            offset = np.zeros(self.h.ndim, dtype=int)
        else:
            offset = np.asarray(offset, dtype=int)
        self.offset = 2 * (self.nh // 2 - offset)
        pad = [(0, 0) for _ in range(self.h.ndim)]
        dopad = False
        for inh, nh in enumerate(self.nh):
            if nh % 2 == 0:
                self.offset[inh] -= 1
            if self.offset[inh] != 0:
                pad[inh] = [
                    self.offset[inh] if self.offset[inh] > 0 else 0,
                    -self.offset[inh] if self.offset[inh] < 0 else 0,
                ]
                dopad = True
        if dopad:
            ncp = get_array_module(h)
            self.h = ncp.pad(self.h, pad, mode="constant")
        self.nh = self.h.shape

        # find out which directions are used for convolution and define offsets
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
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        return self.convolve(x, self.h, mode="same", method=self.method)

    @reshaped
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        return self.correlate(x, self.h, mode="same", method=self.method)
