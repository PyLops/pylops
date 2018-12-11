import numpy as np
from pylops import LinearOperator


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply second-order second derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    Notes
    -----
    The SecondDerivative operator applies a second derivative to any chosen direction of a
    multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered first derivative is:

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / dx

    """
    def __init__(self, N, dims=None, dir=0, sampling=1, dtype='float32'):
        self.N = N
        self.dir = dir
        self.sampling = sampling
        if dims is None:
            self.dims = [self.N, 1]
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = dims
                self.reshape = True
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            y = np.zeros(self.N)
            y[1:-1] = (x[2:]-2*x[1:-1]+x[0:-2])/self.sampling
            # dealing with edges
            #y[0] = (x[1]-2*x[0])/self.sampling
            #y[-1] = (x[-2]-2*x[-1])/self.sampling
        else:
            x = np.reshape(x, (self.dims))
            y = np.zeros((self.dims))
            if self.dir > 0:  # need to bring the dimension to derive to first dimension
                x = np.swapaxes(x, self.dir, 0)
                y = np.swapaxes(y, self.dir, 0)
            y[1:-1] = (x[2:]-2*x[1:-1]+x[0:-2])/self.sampling
            # dealing with edges
            #y[0] = (x[1]-2*x[0])/self.sampling
            #y[-1] = (x[-2]-2*x[-1])/self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            y = np.zeros(self.N)
            y[0:-2] = y[0:-2] + (x[1:-1])/self.sampling
            y[1:-1] = y[1:-1] - (2*x[1:-1])/self.sampling
            y[2:] = y[2:]   + (x[1:-1])/self.sampling
            # dealing with edges
            #y[0] = y[0]  - (2*x[0])/self.sampling
            #y[1] = y[1]  + (x[0])/self.sampling
            #y[-1] = y[-1] - (2*x[-1])/self.sampling
            #y[-2] = y[-2] + (x[-1])/self.sampling
        else:
            x = np.reshape(x, self.dims)
            y = np.zeros((self.dims))
            if self.dir > 0:  # need to bring the dimension to derive to first dimension
                x = np.swapaxes(x, self.dir, 0)
                y = np.swapaxes(y, self.dir, 0)
            y[0:-2] = y[0:-2] + (x[1:-1])/self.sampling
            y[1:-1] = y[1:-1] - (2*x[1:-1])/self.sampling
            y[2:] = y[2:, :] + (x[1:-1])/self.sampling
            # dealing with edges
            #y[0] = y[0]  - (2*x[0])/self.sampling
            #y[1] = y[1]  + (x[0])/self.sampling
            #y[-1] = y[-1] - (2*x[-1])/self.sampling
            #y[-2] = y[-2] + (x[-1])/self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = np.ndarray.flatten(y)
        return y

