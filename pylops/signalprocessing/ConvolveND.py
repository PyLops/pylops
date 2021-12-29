import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import (
    get_array_module,
    get_convolve,
    get_correlate,
    to_cupy_conditional,
)


class ConvolveND(LinearOperator):
    r"""ND convolution operator.

    Apply n-dimensional convolution with a compact filter to model
    (and data) along a set of directions ``dirs`` of a n-dimensional
    array.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model
    h : :obj:`numpy.ndarray`
        nd compact filter to be convolved to input signal
    dims : :obj:`list`
        Number of samples for each dimension
    offset : :obj:`tuple`, optional
        Indices of the center of the compact filter
    dirs : :obj:`tuple`, optional
        Directions along which convolution is applied
        (set to ``None`` for filter of same dimension as input vector)
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct`` or ``fft``).
    dtype : :obj:`str`, optional
        Type of elements in input array.

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
        self, N, h, dims, offset=None, dirs=None, method="fft", dtype="float64"
    ):
        ncp = get_array_module(h)
        self.h = h
        self.nh = np.array(self.h.shape)
        self.dirs = np.arange(len(dims)) if dirs is None else np.array(dirs)

        # padding
        if offset is None:
            offset = np.zeros(self.h.ndim, dtype=int)
        else:
            offset = np.array(offset, dtype=int)
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
            self.h = ncp.pad(self.h, pad, mode="constant")
        self.nh = self.h.shape

        # find out which directions are used for convolution and define offsets
        if len(dims) != len(self.nh):
            dimsh = np.ones(len(dims), dtype=int)
            for idir, dir in enumerate(self.dirs):
                dimsh[dir] = self.nh[idir]
            self.h = self.h.reshape(dimsh)

        if np.prod(dims) != N:
            raise ValueError("product of dims must equal N!")
        else:
            self.dims = np.array(dims)
            self.reshape = True

        # convolve and correate functions
        self.convolve = get_convolve(h)
        self.correlate = get_correlate(h)
        self.method = method

        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        x = np.reshape(x, self.dims)
        y = self.convolve(x, self.h, mode="same", method=self.method)
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        # correct type of h if different from x and choose methods accordingly
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convolve = get_convolve(self.h)
            self.correlate = get_correlate(self.h)
        x = np.reshape(x, self.dims)
        y = self.correlate(x, self.h, mode="same", method=self.method)
        y = y.ravel()
        return y
