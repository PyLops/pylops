__all__ = ["ToCupy"]

from typing import Union

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import to_cupy, to_numpy
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class ToCupy(LinearOperator):
    r"""Convert to CuPy.

    Convert an input NumPy array to CuPy in forward mode,
    and convert back to NumPy in adjoint mode.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    Notes
    -----
    The ToCupy operator is a special operator that does not perform
    any transformation on the input arrays other than converting
    them from NumPy to CuPy. This operator can be used when one
    is interested to create a chain of operators where only one
    (or some of them) act on CuPy arrays, whilst other operate
    on NumPy arrays.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    def _matvec(self, x: NDArray) -> NDArray:
        return to_cupy(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return to_numpy(x)
