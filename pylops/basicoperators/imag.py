__all__ = ["Imag"]

from typing import Union

import numpy as np

from pylops.linearoperator import BaseLinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Imag(BaseLinearOperator):
    r"""Imag operator.

    Return the imaginary component of the input as a real value.
    The adjoint returns a complex number with zero real component and
    the imaginary component set to the real component of the input.

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

        y_{i} = \Im\{x_{i}\} \quad \forall i=0,\ldots,N-1

    In adjoint mode:

    .. math::

        x_{i} = 0 + i\Re\{y_{i}\} \quad \forall i=0,\ldots,N-1

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        dtype: DTypeLike = "complex128",
        name: str = "I",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=dims, clinear=False, name=name
        )
        self.rdtype = np.real(np.ones(1, self.dtype)).dtype

    def _matvec(self, x: NDArray) -> NDArray:
        return x.imag.astype(self.rdtype)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return (0 + 1j * x.real).astype(self.dtype)
