import warnings

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_array


class Flip(LinearOperator):
    r"""Flip along an axis.

    Flip a multi-dimensional array along ``axis``.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which model is flipped.
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
    The Flip operator flips the input model (and data) along any chosen
    direction. For simplicity, given a one dimensional array,
    in forward mode this is equivalent to:

    .. math::
        y[i] = x[N-1-i] \quad \forall i=0,1,2,\ldots,N-1

    where :math:`N` is the dimension of the input model along ``axis``. As this operator is
    self-adjoint, :math:`x` and :math:`y` in the equation above are simply
    swapped in adjoint mode.

    """

    def __init__(self, dims, axis=-1, dtype="float64"):
        self.dims = _value_or_list_like_to_array(dims)
        self.axis = axis
        N = np.prod(self.dims)
        self.shape = (N, N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = np.flip(x, axis=self.axis)
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        return self._matvec(x)
