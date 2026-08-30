"""Skeleton for a new PyLops operator.

Copy into ``pylops/<subpackage>/<operatorname>.py``, rename, and fill in.
Delete anything that does not apply.
"""

__all__ = ["MyOperator"]

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, to_cupy_conditional
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class MyOperator(LinearOperator):
    r"""One-line summary of the operator.

    Longer description of what the operator applies in forward mode and what
    its adjoint does.

    .. versionadded:: X.Y.Z

    Parameters
    ----------
    param : :obj:`numpy.ndarray`
        Description of the main parameter.
    dims : :obj:`list` or :obj:`int`, optional
        Number of samples for each dimension of the model.
    axis : :obj:`int`, optional
        Axis along which the operator is applied.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape.
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``).

    Raises
    ------
    ValueError
        If ``param`` has incompatible size with ``dims``.

    Notes
    -----
    In forward mode the operator applies

    .. math::
        y_i = \ldots \quad \forall i=1,2,\ldots,N

    and in adjoint mode

    .. math::
        x_i = \ldots \quad \forall i=1,2,\ldots,M

    """

    def __init__(
        self,
        param: NDArray,
        dims: int | InputDimsLike | None = None,
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "M",
    ) -> None:
        self.param = param
        self.axis = axis
        dims = param.shape if dims is None else _value_or_sized_to_tuple(dims)
        # dimsd is the shape of the data (output of the forward)
        dimsd = dims
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if type(self.param) is not type(x):
            self.param = to_cupy_conditional(x, self.param)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        # ... forward implementation, y = A x
        return y

    @reshaped
    def _rmatvec(self, y: NDArray) -> NDArray:
        ncp = get_array_module(y)
        if type(self.param) is not type(y):
            self.param = to_cupy_conditional(y, self.param)
        x = ncp.zeros(self.dims, dtype=self.dtype)
        # ... adjoint implementation, x = A^H y (conjugate for complex params!)
        return x
