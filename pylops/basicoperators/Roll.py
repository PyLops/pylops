import numpy as np
from pylops import LinearOperator


class Roll(LinearOperator):
    r"""Roll along an axis.

    Roll a multi-dimensional array along a specified direction ``dir`` for
    a chosen number of samples (``shift``).

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which rolling is applied.
    shift : :obj:`int`, optional
        Number of samples by which elements are shifted
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    The Roll operator is a thin wrapper around :func:`numpy.roll` and shifts
    elements in a multi-dimensional array along a specified direction for a
    chosen number of samples.

    """
    def __init__(self, N, dims=None, dir=0, shift=1, dtype='float64'):
        self.N = N
        self.dir = dir
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N')
            else:
                self.dims = dims
                self.reshape = True
        self.shift = shift
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        y = np.roll(x, shift=self.shift, axis=self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        y = np.roll(x, shift=-self.shift, axis=self.dir)
        return y.ravel()
