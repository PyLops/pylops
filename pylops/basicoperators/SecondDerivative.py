import numpy as np
from pylops import LinearOperator


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply second-order second derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    The SecondDerivative operator applies a second derivative to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / dx^2

    """
    def __init__(self, N, dims=None, dir=0, sampling=1,
                 edge=False, dtype='float64'):
        self.N = N
        self.sampling = sampling
        self.edge = edge
        if dims is None:
            self.dims = (self.N, )
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = dims
                self.reshape = True
        self.dir = dir if dir >= 0 else len(self.dims) + dir
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[1:-1] = (x[2:] - 2*x[1:-1] + x[0:-2]) / self.sampling**2
            if self.edge:
                y[0] = (x[0] - 2*x[1] + x[2]) / self.sampling**2
                y[-1] = (x[-3] - 2*x[-2] + x[-1]) / self.sampling**2
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[1:-1] = (x[2:] - 2*x[1:-1] + x[0:-2])/self.sampling**2
            if self.edge:
                y[0] = (x[0] - 2*x[1] + x[2]) / self.sampling ** 2
                y[-1] = (x[-3] - 2*x[-2] + x[-1]) / self.sampling ** 2
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[0:-2] += (x[1:-1]) / self.sampling**2
            y[1:-1] -= (2*x[1:-1]) / self.sampling**2
            y[2:] += (x[1:-1]) / self.sampling**2
            if self.edge:
                y[0] += x[0] / self.sampling**2
                y[1] -= 2 * x[0] / self.sampling**2
                y[2] += x[0] / self.sampling ** 2
                y[-3] += x[-1] / self.sampling**2
                y[-2] -= 2 * x[-1] / self.sampling**2
                y[-1] += x[-1] / self.sampling**2
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[0:-2] += (x[1:-1]) / self.sampling**2
            y[1:-1] -= (2*x[1:-1]) / self.sampling**2
            y[2:] += (x[1:-1]) / self.sampling**2
            if self.edge:
                y[0] += x[0] / self.sampling ** 2
                y[1] -= 2 * x[0] / self.sampling ** 2
                y[2] += x[0] / self.sampling ** 2
                y[-3] += x[-1] / self.sampling ** 2
                y[-2] -= 2 * x[-1] / self.sampling ** 2
                y[-1] += x[-1] / self.sampling ** 2
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
