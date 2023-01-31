__all__ = ["DiscreetCosine"]

from typing import Optional, Union

import numpy as np
from scipy import fft

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray
from pylops.utils.decorators import reshaped

class DiscreetCosine(LinearOperator):
    r"""Discreet Cosine Transform

    Performs discreet cosine transform on the given multi-dimensional
    array along the given axis.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        Axes over which the DCT is computed. 
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
    The DiscreetCosine is implemented in normalization mode = "ortho" to make the scaling symmetrical.
    It is a type 2 DCT transform.

    To calculate the DCT we use the following
    
    .. math::
       f = \begin{cases}
       \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
       \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

    which makes the corresponding matrix of coefficients orthonormal (``O @ O.T = np.eye(N)``).
    """

    def __init__(
        self, 
        dims: Union[int, InputDimsLike], 
        axes: int = None,
        dtype: DTypeLike = "float64",
        name: str = "C"
    ) -> None:
        self.axes = axes
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return fft.dctn(x, axes=self.axes, norm="ortho")
        

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        return fft.idctn(x, axes=self.axes, norm="ortho")
