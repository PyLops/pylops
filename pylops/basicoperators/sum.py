__all__ = ["Sum"]

import numpy as np

from pylops.linearoperator import BaseLinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Sum(BaseLinearOperator):
    r"""Sum operator.

    Sum along ``axis`` of a multi-dimensional
    array (at least 2 dimensions are required) in forward model, and spread
    along the same axis in adjoint mode.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which model is summed.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

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
    of size ``dims`` and sums values along ``axis``:

    .. math::

        y_j = \sum_i x_{i, j}

    In adjoint mode, the data is spread along the same direction:

    .. math::

        x_{i, j} = y_j   \quad \forall i=0, N-1

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        # to avoid reducing matvec to a scalar
        dims = (dims[0], 1) if len(dims) == 1 else dims
        self.axis = axis
        # data dimensions
        dimsd = list(dims).copy()
        dimsd.pop(self.axis)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # array of ones with dims of model in self.axis for np.tile in adjoint mode
        self.tile = np.ones(len(self.dims), dtype=int)
        self.tile[self.axis] = self.dims[self.axis]

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return x.sum(axis=self.axis)

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.expand_dims(x, self.axis)
        y = ncp.tile(y, self.tile)
        return y
