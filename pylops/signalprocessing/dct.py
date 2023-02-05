__all__ = ["DCT"]

from typing import List, Optional, Union

import numpy as np
from scipy import fft

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DCT(LinearOperator):
    r"""Discrete Cosine Transform.

    Apply 1D or ND-Cosine Transform along one or more ``axes`` of a multi-dimensional
    array of size ``dims``.

    This operator is an overload of :func:`scipy.fft.dctn` in forward mode and :func:`scipy.fft.idctn`
    in adjoint mode.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
    type : :obj:`int`, optional
        Type of DCT (see scipy's documentation for more details). Default type is 2.
    axes : :obj:`int` or :obj:`list`, optional
        Axes over which the DCT is computed. If ``None``, the transform is applied
        over all axes.
    workers :obj:`int`, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from os.cpu_count().
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
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
        If ``type`` is different from 1, 2, 3, or 4.

    Notes
    -----
    The DCT operator applies the Discrete Cosine Transform in forward mode and the Inverse Discrete Cosine Transform
    in adjoint mode. This transform expresses a signal as a sum of cosine functions oscillating at different
    frequencies. By doing so, no information is lost and the energy is compacted into the top left corner of the
    transform. When applied to multi-dimensional arrays, the DCT operator is simply a cascade of one-dimensional DCT
    operators acting along the different axes,

    Finally, note that the DCT operator is implemented with normalization mode ``norm="ortho"`` to ensure symmetric
    scaling.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        type: int = 2,
        axes: Union[int, List[int]] = None,
        dtype: DTypeLike = "float64",
        workers: Optional[int] = None,
        name: str = "C",
    ) -> None:

        if type > 4 or type < 1:
            raise ValueError("wrong type value, it can only be 1, 2, 3 or 4")
        self.type = type
        self.axes = axes
        self.workers = workers
        self.dims = _value_or_sized_to_tuple(dims)
        super().__init__(
            dtype=np.dtype(dtype), dims=self.dims, dimsd=self.dims, name=name
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return fft.dctn(
            x, axes=self.axes, type=self.type, norm="ortho", workers=self.workers
        )

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        return fft.idctn(
            x, axes=self.axes, type=self.type, norm="ortho", workers=self.workers
        )
