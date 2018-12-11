import numpy as np
from pylops import LinearOperator


class FirstDerivative(LinearOperator):
    r"""First derivative.

    Apply second-order centered first derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
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
    The FirstDerivative operator applies a first derivative to any chosen direction of a
    multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered first derivative is:

    .. math::
        y[i] = (0.5x[i+1] - 0.5x[i-1]) / dx

    """
    def __init__(self, N, dims=None, dir=0, sampling=1., dtype='float32'):
        self.N = N
        self.dir = dir
        self.sampling = sampling
        if dims is None:
            self.dims = (self.N, )
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N')
            else:
                self.dims = dims
                self.reshape = True
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        """ Apply forward first order derivative (y=A*x)

        :param np.ndarray x: vector

        :return: A*x
        :rtype np.ndarray
        """
        if not self.reshape:
            y = np.zeros(self.N, self.dtype)
            y[1:-1] = (0.5*x[2:]-0.5*x[0:-2])/self.sampling
        else:
            x = np.reshape(x, self.dims)
            y = np.zeros(self.dims, self.dtype)
            if self.dir > 0:  # need to bring the dimension to derive to first dimension
                x = np.swapaxes(x, self.dir, 0)
                y = np.swapaxes(y, self.dir, 0)
            y[1:-1] = (0.5*x[2:]-0.5*x[0:-2])/self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.flatten()
        return y

    def _rmatvec(self, x):
        """ Apply adjoint first order derivative (x=A'*y)

        :param np.ndarray x: vector

        :return: A'*x
        :rtype np.ndarray
        """
        if not self.reshape:
            y = np.zeros(self.N, self.dtype)
            y[0:-2] = y[0:-2] - (0.5*x[1:-1])/self.sampling
            y[2:] = y[2:] + (0.5*x[1:-1])/self.sampling
        else:
            x = np.reshape(x, (self.dims))
            y = np.zeros((self.dims), self.dtype)
            if self.dir > 0:  # need to bring the dimension to derive to first dimension
                x = np.swapaxes(x, self.dir, 0)
                y = np.swapaxes(y, self.dir, 0)
            y[0:-2] = y[0:-2] - (0.5 * x[1:-1])/self.sampling
            y[2:] = y[2:] + (0.5 * x[1:-1])/self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.flatten()
        return y
