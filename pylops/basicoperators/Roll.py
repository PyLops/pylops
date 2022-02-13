import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_array


class Roll(LinearOperator):
    r"""Roll along an axis.

    Roll a multi-dimensional array along a specified direction ``dir`` for
    a chosen number of samples (``shift``).

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
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

    def __init__(self, dims, dir=0, shift=1, dtype="float64"):
        self.dims = _value_or_list_like_to_array(dims)
        N = np.prod(self.dims)
        self.dir = dir
        self.shift = shift
        self.shape = (N, N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = np.roll(x, shift=self.shift, axis=self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims)
        y = np.roll(x, shift=-self.shift, axis=self.dir)
        return y.ravel()
