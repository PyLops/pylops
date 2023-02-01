__all__ = ["Transpose"]

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops.linearoperator import BaseLinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Transpose(BaseLinearOperator):
    r"""Transpose operator.

    Transpose axes of a multi-dimensional array. This operator works with
    flattened input model (or data), which are however multi-dimensional in
    nature and will be reshaped and treated as such in both forward and adjoint
    modes.

    Parameters
    ----------
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Direction along which transposition is applied
    dtype : :obj:`str`, optional
        Type of elements in input array
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

    Raises
    ------
    ValueError
        If ``axes`` contains repeated dimensions (or a dimension is missing)

    Notes
    -----
    The Transpose operator reshapes the input model into a multi-dimensional
    array of size ``dims`` and transposes (or swaps) its axes as defined
    in ``axes``.

    Similarly, in adjoint mode the data is reshaped into a multi-dimensional
    array whose size is a permuted version of ``dims`` defined by ``axes``.
    The array is then rearragned into the original model dimensions ``dims``.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike,
        dtype: DTypeLike = "float64",
        name: str = "T",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        ndims = len(dims)
        self.axes = [normalize_axis_index(ax, ndims) for ax in axes]

        # find out if all axes are present only once in axes
        if len(np.unique(self.axes)) != ndims:
            raise ValueError("axes must contain each direction once")

        # find out how axes should be transposed in adjoint mode
        axesd = np.empty(ndims, dtype=int)
        axesd[self.axes] = np.arange(ndims, dtype=int)

        dimsd = np.empty(ndims, dtype=int)
        dimsd[axesd] = dims
        self.axesd = list(axesd)

        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return x.transpose(self.axes)

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        return x.transpose(self.axesd)
