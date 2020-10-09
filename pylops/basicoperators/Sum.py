import numpy as np
from pylops import LinearOperator


class Sum(LinearOperator):
    r"""Sum operator.

    Sum along an axis of a multi-dimensional
    array (at least 2 dimensions are required) in forward model, and spread
    along the same axis in adjoint mode.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dir : :obj:`int`
        Direction along which summation is performed.
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
    Given a two dimensional array, the *Sum* operator re-arranges
    the input model into a multi-dimensional array
    of size ``dims`` and sums values along direction ``dir``:

    .. math::

        y_j = \sum_i x_{i, j}

    In adjoint mode, the data is spread along the same direction:

    .. math::

        x_{i, j} = y_j   \quad \forall i=0, N-1

    """
    def __init__(self, dims, dir, dtype='float64'):
        if len(dims) == 1:
            dims = (dims[0], 1) # to avoid reducing matvec to a scalar
        self.dims = dims
        self.dir = dir
        # data dimensions
        self.dims_d = list(dims).copy()
        self.dims_d.pop(dir)
        # array of ones with dims of model in dir for np.tile in adjoint mode
        self.tile = np.ones(len(dims), dtype=np.int)
        self.tile[dir] = self.dims[dir]
        self.dtype = np.dtype(dtype)
        self.shape = (np.prod(self.dims_d), np.prod(dims))
        self.explicit = False

    def _matvec(self, x):
        y = x.reshape(self.dims)
        y = np.sum(y, axis=self.dir)
        return y.flatten()

    def _rmatvec(self, x):
        y = x.reshape(self.dims_d)
        y = np.expand_dims(y, self.dir)
        y = np.tile(y, self.tile)
        return y.flatten()
