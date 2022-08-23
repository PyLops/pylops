__all__ = ["Roll"]

from typing import List, Union

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.decorators import reshaped


class Roll(LinearOperator):
    r"""Roll along an axis.

    Roll a multi-dimensional array along ``axis`` for
    a chosen number of samples (``shift``).

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which model is rolled.
    shift : :obj:`int`, optional
        Number of samples by which elements are shifted
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
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    The Roll operator is a thin wrapper around :func:`numpy.roll` and shifts
    elements in a multi-dimensional array along a specified direction for a
    chosen number of samples.

    """

    def __init__(
        self,
        dims: Union[int, List[int]],
        axis: int = -1,
        shift: int = 1,
        dtype: str = "float64",
        name: str = "R",
    ) -> None:
        dims = _value_or_list_like_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)
        self.axis = axis
        self.shift = shift

    @reshaped(swapaxis=True)
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.roll(x, shift=self.shift, axis=-1)

    @reshaped(swapaxis=True)
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.roll(x, shift=-self.shift, axis=-1)
