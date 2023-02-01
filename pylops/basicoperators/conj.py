__all__ = ["Conj"]


from typing import Union

import numpy as np

from pylops.optimization.base_linearoperator import BaseLinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Conj(BaseLinearOperator):
    r"""Complex conjugate operator.

    Return the complex conjugate of the input. It is self-adjoint.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
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
    In forward mode:

    .. math::

        y_{i} = \Re\{x_{i}\} - i\Im\{x_{i}\} \quad \forall i=0,\ldots,N-1

    In adjoint mode:

    .. math::

        x_{i} = \Re\{y_{i}\} - i\Im\{y_{i}\} \quad \forall i=0,\ldots,N-1

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        dtype: DTypeLike = "complex128",
        name: str = "C",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=dims, clinear=False, name=name
        )

    def _matvec(self, x: NDArray) -> NDArray:
        return x.conj()

    def _rmatvec(self, x: NDArray) -> NDArray:
        return x.conj()
