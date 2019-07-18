import numpy as np
from scipy.signal import convolve, correlate
from pylops import LinearOperator


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
    def __init__(self, N, h, dims, offset=(0, 0, 0), dirs=None,
                 method='fft', dtype='float64'):
        self.h = np.array(h)
        self.nh = np.array(self.h.shape)
        self.dirs = np.arange(len(dims)) if dirs is None else np.array(dirs)

        # find out which directions are used for convolution and define offsets
        if len(dims) != len(self.nh):
            self.offset = self.nh // 2
            dimsh = np.ones(len(dims), dtype=np.int)
            for dir in self.dirs:
                dimsh[dir] = self.nh[dir]
                self.offset[dir] = int(offset[dir])
            self.h = self.h.reshape(dimsh)
        else:
            self.offset = np.array(offset).astype(np.int)
        for dir in self.dirs:
            self.offset[dir] = int(offset[dir])

        # padding
        self.offset = 2 * (self.nh // 2 - self.offset)
        pad = [(0, 0) for _ in range(len(dims))]
        dopad = False
        for inh, nh in enumerate(self.nh):
            if nh % 2 == 0:
                self.offset[inh] -= 1
            if self.offset[inh] != 0:
                pad[inh] = [self.offset[inh] if self.offset[inh] > 0 else 0,
                            -self.offset[inh] if self.offset[inh] < 0 else 0]
                dopad = True
        if dopad:
            self.h = np.pad(self.h, pad, mode='constant')

        if np.prod(dims) != N:
            raise ValueError('product of dims must equal N!')
        else:
            self.dims = np.array(dims)
            self.reshape = True
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.method = method
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = convolve(x, self.h, mode='same', method=self.method)
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims)
        y = correlate(x, self.h, mode='same', method=self.method)
        y = y.ravel()
        return y
