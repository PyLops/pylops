import numpy as np
from scipy.signal import convolve2d, correlate2d
from pylops import LinearOperator


class Convolve2D(LinearOperator):
    r"""2D convolution operator.

    Apply two-dimensional convolution with a compact filter to model (and data) along
    a pair of specific directions of a two or three-dimensional array depending
    on the choice of ``dirs``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model
    h : :obj:`numpy.ndarray`
        2d compact filter to be convolved to input signal
    dims : :obj:`list`
        Number of samples for each dimension
    offset : :obj:`tuple`, optional
        Indeces of the center of the compact filter
    nodir : :obj:`int`, optional
        Direction along which convolution is NOT applied (set to None for 2d arrays)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (True) or not (False)

    Notes
    -----
    The Convolve2D operator applies two-dimensional convolution between the input signal
    :math:`d(t,x)` and a compact filter kernel :math:`h(t,x)` in forward model:

    .. math::
        y(t,x) = \int_{-\inf}^{\inf}\int_{-\inf}^{\inf} h(t-\tau,x-\chi) d(\tau,\chi) d\tau d\chi

    This operation can be discretized as follows

    .. math::
        y[i,n] = \sum_{j=-\inf}^{\inf} \sum_{m=-\inf}^{\inf} h[i-j,n-m] d[j,m]


    as well as performed in the frequency domain.

    .. math::
        Y(f, k_x) = \mathscr{F} (h(t,x)) * \mathscr{F} (d(t,x))

    Convolve2D operator uses :py:func:`scipy.signal.convolve2d` that automatically chooses
    the best domain for the operation to be carried out.

    As the adjoint of convolution is correlation, Convolve2D operator applies correlation
    in the adjoint mode.

    In time domain:

    .. math::
        y(t,x) = \int_{-\inf}^{\inf}\int_{-\inf}^{\inf} h(t+\tau,x+\chi) d(\tau,\chi) d\tau d\chi

    or in frequency domain:

    .. math::
        y(t, x) = \mathscr{F}^{-1} (H(f, k_x)^* * X(f, k_x))

    """
    def __init__(self, N, h, dims, offset=(0, 0), nodir=None, dtype=None):
        self.offset = np.array(offset, dtype=np.int)
        self.h = np.array(h)
        self.nodir = nodir

        if np.prod(dims) != N:
            raise ValueError('product of dims must equal N!')
        else:
            self.dims = np.array(dims)
            self.reshape = True
        if self.nodir is None:
            #self.shape = (np.prod(self.dims+self.h.shape-1-2*self.offset), np.prod(self.dims))
            self.shape = (np.prod(self.dims), np.prod(self.dims))
        else:
            #self.shape = (np.prod(np.delete(self.dims, self.nodir)+self.h.shape-1-2*self.offset)*self.dims[self.nodir],
            #              np.prod(self.dims))
            self.shape = (np.prod(self.dims), np.prod(self.dims))

        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.nodir is None:
            y = convolve2d(x, self.h, mode='full')
            y = y[self.offset[0]:-self.h.shape[0]+self.offset[0]+1,
                  self.offset[1]:-self.h.shape[1]+self.offset[1]+1]
        else:
            x = np.swapaxes(x, self.nodir, 0)
            y = np.array([convolve2d(x[i], self.h, mode='full')
                          for i in range(self.dims[self.nodir])])
            y = y[:, self.offset[0]:-self.h.shape[0]+self.offset[0]+1,
                     self.offset[1]:-self.h.shape[1]+self.offset[1]+1]
            y = np.swapaxes(y, self.nodir, 0)
        y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims)
        if self.nodir is None:
            y = correlate2d(x, self.h, mode='full')
            y = y[self.h.shape[0]-self.offset[0]-1:-self.offset[0],
                  self.h.shape[1]-self.offset[1]-1:-self.offset[1]]
        else:
            x = np.swapaxes(x, self.nodir, 0)
            y = np.array([correlate2d(x[i], self.h, mode='full') for i in range(self.dims[self.nodir])])
            y = y[:, self.h.shape[0]-self.offset[0]-1:-self.offset[0],
                     self.h.shape[1]-self.offset[1]-1:-self.offset[1]]
            y = np.swapaxes(y, self.nodir, 0)
        y = np.ndarray.flatten(y)
        return y
