from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple


class Imag(LinearOperator):
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
        dims: Union[int, Tuple[int]],
        dtype: str = "complex128",
        name: str = "I",
    ) -> None:
        dims = _value_or_list_like_to_tuple(dims)
        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=dims, clinear=False, name=name
        )
        self.rdtype = np.real(np.ones(1, self.dtype)).dtype

    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x.imag.astype(self.rdtype)

    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (0 + 1j * x.real).astype(self.dtype)
