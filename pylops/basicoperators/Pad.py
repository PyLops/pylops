from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.decorators import reshaped


class Pad(LinearOperator):
    r"""Pad operator.

    Zero-pad model in forward model and extract non-zero subsequence
    in adjoint. Padding can be performed in one or multiple directions to any
    multi-dimensional input arrays.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
    pad : :obj:`tuple`
        Number of samples to pad. If ``dims`` is a scalar, ``pad`` is a single
        tuple ``(pad_in, pad_end)``. If ``dims`` is a tuple,
        ``pad`` is a tuple of tuples where each inner tuple contains
        the number of samples to pad in each dimension
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

    Raises
    ------
    ValueError
        If any element of ``pad`` is negative.

    Notes
    -----
    Given an array of size :math:`N`, the *Pad* operator simply adds
    :math:`\text{pad}_\text{in}` at the start and :math:`\text{pad}_\text{end}` at the end in forward mode:

    .. math::

        y_{i} = x_{i-\text{pad}_\text{in}}  \quad \forall
        i=\text{pad}_\text{in},\ldots,\text{pad}_\text{in}+N-1

    and :math:`y_i = 0 \quad \forall
    i=0,\ldots,\text{pad}_\text{in}-1, \text{pad}_\text{in}+N-1,\ldots,N+\text{pad}_\text{in}+\text{pad}_\text{end}`

    In adjoint mode, values from :math:`\text{pad}_\text{in}` to :math:`N-\text{pad}_\text{end}` are
    extracted from the data:

    .. math::

        x_{i} = y_{\text{pad}_\text{in}+i}  \quad \forall i=0, N-1

    """

    def __init__(
        self,
        dims: Union[int, Tuple],
        pad: Tuple,
        dtype: str = "float64",
        name: str = "P",
    ) -> None:
        if np.any(np.array(pad) < 0):
            raise ValueError("Padding must be positive or zero")
        dims = _value_or_list_like_to_tuple(dims)
        # Accept (padbeg, padend) and [(padbeg, padend)]
        self.pad = [pad] if len(dims) == 1 and len(pad) == 2 else pad
        dimsd = [dim + before + after for dim, (before, after) in zip(dims, self.pad)]
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

    @reshaped
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.pad(x, self.pad, mode="constant")

    @reshaped
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        for ax, (before, _) in enumerate(self.pad):
            x = np.take(x, np.arange(before, before + self.dims[ax]), axis=ax)
        return x
