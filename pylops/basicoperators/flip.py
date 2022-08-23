__all__ = ["Flip"]

from typing import List, Union

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.decorators import reshaped


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
    The Flip operator flips the input model (and data) along any chosen
    direction. For simplicity, given a one dimensional array,
    in forward mode this is equivalent to:

    .. math::
        y[i] = x[N-1-i] \quad \forall i=0,1,2,\ldots,N-1

    where :math:`N` is the dimension of the input model along ``axis``. As this operator is
    self-adjoint, :math:`x` and :math:`y` in the equation above are simply
    swapped in adjoint mode.

    """

    def __init__(
        self,
        dims: Union[int, List[int]],
        axis: int = -1,
        dtype: str = "float64",
        name: str = "F",
    ) -> None:
        dims = _value_or_list_like_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)
        self.axis = axis

    @reshaped(swapaxis=True)
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        y = np.flip(x, axis=-1)
        return y

    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self._matvec(x)
