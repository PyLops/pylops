import numpy as np
from pylops import LinearOperator


class Symmetrize(LinearOperator):
    r"""Symmetrize along an axis.

    Symmetrize a multi-dimensional array along a specified direction ``dir``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model. Symmetric data has  :math:`2N-1` samples
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which symmetrization is applied
    dtype : :obj:`str`, optional
        Type of elements in input array

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    The Symmetrize operator constructs a symmetric array given an input model
    in forward mode, by pre-pending the input model in reversed order.

    For simplicity, given a one dimensional array, the forward operation can
    be expressed as:

    .. math::
        y[i] = \begin{cases}
        x[i-N],& i\geq N\\
        x[N-i],& \text{otherwise}
        \end{cases}

    for :math:`i=0,1,2,...,2N-2`, where :math:`N` is the lenght of
    the input model.

    In adjoint mode, the Symmetrize operator assigns the sums of the elements
    in position :math:`N-i` and :math:`N+i` to position :math:`i` as follows:

    .. math::
        \begin{multline}
        x[i] = y[N-i]+y[N+i] \quad \forall i=1,2,...,N-1
        \end{multline}

    apart from the central sample where :math:`x[0] = y[N]`.
    """
    def __init__(self, N, dims=None, dir=0, dtype='float64'):
        self.N = N
        self.dir = dir
        if dims is None:
            self.dims = (self.N,)
            self.dimsd = (self.N*2-1,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N')
            else:
                self.dims = dims
                self.dimsd = list(dims)
                self.dimsd[self.dir] = dims[self.dir]*2-1
                self.reshape = True
        self.nsym = self.dims[self.dir]
        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros(self.dimsd, dtype=self.dtype)
        if self.reshape:
            x = np.reshape(x, self.dims)
        if self.dir > 0:# bring the dimension to symmetrize to first
            x = np.swapaxes(x, self.dir, 0)
            y = np.swapaxes(y, self.dir, 0)
        y[self.nsym-1:] = x
        y[:self.nsym-1] = x[-1:0:-1]
        if self.dir > 0:
            y = np.swapaxes(y, 0, self.dir)
        if self.reshape:
            y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dimsd)
        if self.dir > 0:  # bring the dimension to symmetrize to first
            x = np.swapaxes(x, self.dir, 0)
        y = x[self.nsym-1:].copy()
        y[1:] += x[self.nsym-2::-1]
        if self.dir > 0:
            y = np.swapaxes(y, 0, self.dir)
        if self.reshape:
            y = np.ndarray.flatten(y)
        return y
