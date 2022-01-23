import warnings

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module


class Sum(LinearOperator):
    r"""Sum operator.

    Sum along ``axis`` of a multi-dimensional
    array (at least 2 dimensions are required) in forward model, and spread
    along the same axis in adjoint mode.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0

        Axis along which model is summed.
    dir : :obj:`int`, optional

        .. deprecated:: 2.0
            Use ``axis`` instead.

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

    def __init__(self, dims, axis=-1, dir=None, dtype="float64"):
        if len(dims) == 1:
            dims = (dims[0], 1)  # to avoid reducing matvec to a scalar
        self.dims = dims
        if dir is not None:
            warnings.warn(
                "dir is deprecated in version 2.0, use axis instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            self.axis = dir
        else:
            self.axis = axis
        # data dimensions
        self.dims_d = list(dims).copy()
        self.dims_d.pop(self.axis)
        # array of ones with dims of model in self.axis for np.tile in adjoint mode
        self.tile = np.ones(len(dims), dtype=int)
        self.tile[self.axis] = self.dims[self.axis]
        self.dtype = np.dtype(dtype)
        self.shape = (np.prod(self.dims_d), np.prod(dims))
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        y = x.reshape(self.dims)
        y = ncp.sum(y, axis=self.axis)
        return y.ravel()

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        y = x.reshape(self.dims_d)
        y = ncp.expand_dims(y, self.axis)
        y = ncp.tile(y, self.tile)
        return y.ravel()
