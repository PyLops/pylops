__all__ = ["DCT"]

from typing import List, Optional, Union

import numpy as np
from scipy import fft

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DCT(LinearOperator):
    r"""Discrete Cosine Transform

    Performs Discrete cosine transform on the given multi-dimensional
    array along the given axis.
    It uses the ``scipy.fft.dctn`` for forward mode and ``scipy.fft.idctn`` for adjoint mode.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    type : :obj:`int`, optional
        Type of the DCT (see Notes). Default type is 2.
    axes : :obj:`list` or :obj:`int`, optional
        Axes over which the DCT is computed. If not given, the last len(dims) axes are used,
        or all axes if dims is also not specified.
    workers :obj:`int`, optional
        Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count().
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

    Notes
    -----
    The DCT is implemented in normalization mode = "ortho" to make the scaling symmetrical.
    The cosines are normalized to express the real part of the orthonormal fourier transform.
    This allows arbitrary functions to be expressed exactly. No information is lost by taking the DCT and
    the energy is compacted into the top left corner of the transform.

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
            raise ValueError("wrong value of type it can only be 1, 2, 3 or 4")
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
