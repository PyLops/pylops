import numpy as np
import numpy.ma as np_ma

from pylops import LinearOperator


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    M : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of available samples for data selection.
    dims : :obj:`list`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which restriction is applied.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    See Also
    --------
    pylops.signalprocessing.Interp : Interpolation operator

    Notes
    -----
    Extraction (or *sampling*) of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = x_{l_i}  \quad \forall i=1,2,...,N

    where :math:`\mathbf{l}=[l_1, l_2,..., l_N]` is a vector containing the indeces
    of the original array at which samples are taken.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`\mathbf{l}=[l_1, l_2,..., l_M]` in the model vector:

    .. math::

        x_{l_i} = y_i  \quad \forall i=1,2,...,N

    and :math:`x_{j}=0 j \neq l_i` (i.e., at all other locations in input
    vector).

    """
    def __init__(self, M, iava, dims=None, dir=0,
                 dtype='float64', inplace=True):
        self.M = M
        self.dir = dir
        self.iava = iava
        if dims is None:
            self.N = len(iava)
            self.dims = (self.M, )
            self.reshape = False
        else:
            if np.prod(dims) != self.M:
                raise ValueError('product of dims must equal M!')
            else:
                self.dims = dims # model dimensions
                self.dimsd = list(dims) # data dimensions
                self.dimsd[self.dir] = len(iava)
                self.iavareshape = [1] * self.dir + [len(self.iava)] + \
                                   [1] * (len(self.dims) - self.dir - 1)
                self.N = np.prod(self.dimsd)
                self.reshape = True
        self.inplace = inplace
        self.shape = (self.N, self.M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.inplace: x = x.copy()
        if not self.reshape:
            y = x[self.iava]
        else:
            x = np.reshape(x, self.dims)
            y = np.take(x, self.iava, axis=self.dir)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        if not self.inplace: x = x.copy()
        if not self.reshape:
            y = np.zeros(self.dims, dtype=self.dtype)
            y[self.iava] = x
        else:
            x = np.reshape(x, self.dimsd)
            y = np.zeros(self.dims, dtype=self.dtype)
            np.put_along_axis(y, np.reshape(self.iava, self.iavareshape),
                              x, axis=self.dir)
            y = y.ravel()
        return y

    def mask(self, x):
        """Apply mask to input signal returning a signal of same size with
        values at ``iava`` locations and ``0`` at other locations

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array (can be either flattened or not)

        Returns
        ----------
        y : :obj:`numpy.ma.core.MaskedArray`
            Masked array.

        """
        y = np_ma.array(np.zeros(self.dims), mask=np.ones(self.dims),
                        dtype=self.dtype)
        if self.reshape:
            x = np.reshape(x, self.dims)
            x = np.swapaxes(x, self.dir, 0)
            y = np.swapaxes(y, self.dir, 0)
        y.mask[self.iava] = False
        y[self.iava] = x[self.iava]
        if self.reshape:
            y = np.swapaxes(y, 0, self.dir)
        return y
