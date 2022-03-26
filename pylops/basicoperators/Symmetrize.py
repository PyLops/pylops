import warnings

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple, reshape_flatten
from pylops.utils.backend import get_array_module


class Symmetrize(LinearOperator):
    r"""Symmetrize along an axis.

    Symmetrize a multi-dimensional array along ``axis``.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which model is symmetrized.
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
        x[i-N+1],& i\geq N\\
        x[N-1-i],& \text{otherwise}
        \end{cases}

    for :math:`i=0,1,2,\ldots,2N-2`, where :math:`N` is the dimension of the input
    model.

    In adjoint mode, the Symmetrize operator assigns the sums of the elements
    in position :math:`N-1-i` and :math:`N-1+i` to position :math:`i` as follows:

    .. math::
        \begin{multline}
        x[i] = y[N-1-i]+y[N-1+i] \quad \forall i=0,2,\ldots,N-1
        \end{multline}

    apart from the central sample where :math:`x[0] = y[N-1]`.
    """

    def __init__(self, dims, axis=-1, dtype="float64"):
        self.dims = _value_or_list_like_to_tuple(dims)
        self.axis = axis
        dimsd = list(self.dims)
        dimsd[self.axis] = self.dims[self.axis] * 2 - 1
        self.dimsd = tuple(dimsd)
        self.nsym = self.dims[self.axis]

        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    @reshape_flatten()
    def _matvec(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.swapaxes(y, self.axis, -1)
        y[..., self.nsym - 1 :] = x
        y[..., : self.nsym - 1] = x[..., -1:0:-1]
        y = ncp.swapaxes(y, -1, self.axis)
        return y

    @reshape_flatten(forward=False)
    def _rmatvec(self, x):
        ncp = get_array_module(x)
        x = ncp.swapaxes(x, self.axis, -1)
        y = x[..., self.nsym - 1 :].copy()
        y[..., 1:] += x[..., self.nsym - 2 :: -1]
        y = ncp.swapaxes(y, -1, self.axis)
        return y
