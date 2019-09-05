import numpy as np
from pylops import LinearOperator


class Transpose(LinearOperator):
    r"""Transpose operator.

    Transpose axes of a multi-dimensional array. This operator works with
    flattened input model (or data), which are however multi-dimensional in
    nature and will be reshaped and treated as such in both forward and adjoint
    modes.

    Parameters
    ----------
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Direction along which transposition is applied
    dtype : :obj:`str`, optional
        Type of elements in input array

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``axes`` contains repeated dimensions (or a dimension is missing)

    Notes
    -----
    The Transpose operator reshapes the input model into a multi-dimensional
    array of size ``dims`` and transposes (or swaps) its axes as defined
    in ``axes``.

    Similarly, in adjoint mode the data is reshaped into a multi-dimensional
    array whose size is a permuted version of ``dims`` defined by ``axes``.
    The array is then rearragned into the original model dimensions ``dims``.

    """
    def __init__(self, dims, axes, dtype='float64'):
        self.dims = list(dims)
        self.axes = list(axes)

        # find out if all axes are present only once in axes
        ndims = len(self.dims)
        if len(np.unique(self.axes)) != ndims:
            raise ValueError('axes must contain each direction once')

        # find out how axes should be transposed in adjoint mode
        self.axesd = np.zeros(ndims, dtype=np.int)
        self.dimsd = np.zeros(ndims, dtype=np.int)
        self.axesd[self.axes] = np.arange(ndims, dtype=np.int)
        self.dimsd[self.axesd] = self.dims
        self.axesd = list(self.axesd)

        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = x.reshape(self.dims)
        y = y.transpose(self.axes)
        return y.ravel()

    def _rmatvec(self, x):
        y = x.reshape(self.dimsd)
        y = y.transpose(self.axesd)
        return y.ravel()
