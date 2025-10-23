__all__ = ["Smoothing1D"]

from typing import Union

import numpy as np

from pylops.signalprocessing import Convolve1D
from pylops.utils.typing import DTypeLike, InputDimsLike


class Smoothing1D(Convolve1D):
    r"""1D Smoothing.

    Apply smoothing to model (and data) to a multi-dimensional array
    along ``axis``.

    Parameters
    ----------
    nsmooth : :obj:`int`
        Length of smoothing operator (must be odd, if even it will be
        increased by 1).
    dims : :obj:`tuple` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which model (and data) are smoothed.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    nh : :obj:`int`
        Length of the filter.
    hstar : :obj:`numpy.ndarray`
        Time-reversed filter used in adjoint.
    convfunc : :obj:`callable`
        Function handler used to perform convolution.
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    Notes
    -----
    The Smoothing1D operator is a special type of convolutional operator that
    convolves the input model (or data) with a constant filter of size
    :math:`n_\text{smooth}`:

    .. math::
        \mathbf{f} = [ 1/n_\text{smooth}, 1/n_\text{smooth}, ..., 1/n_\text{smooth} ]

    When applied to the first direction:

    .. math::
        y[i,j,k] = 1/n_\text{smooth} \sum_{l=-(n_\text{smooth}-1)/2}^{(n_\text{smooth}-1)/2}
        x[l,j,k]

    Similarly when applied to the second direction:

    .. math::
        y[i,j,k] = 1/n_\text{smooth} \sum_{l=-(n_\text{smooth}-1)/2}^{(n_\text{smooth}-1)/2}
        x[i,l,k]

    and the third direction:

    .. math::
        y[i,j,k] = 1/n_\text{smooth} \sum_{l=-(n_\text{smooth}-1)/2}^{(n_\text{smooth}-1)/2}
        x[i,j,l]

    Note that since the filter is symmetrical, the *Smoothing1D* operator is
    self-adjoint.

    """

    def __init__(
        self,
        nsmooth: int,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ):
        if nsmooth % 2 == 0:
            nsmooth += 1
        h = np.ones(nsmooth) / float(nsmooth)
        offset = (nsmooth - 1) // 2
        super().__init__(
            dims=dims, h=h, axis=axis, offset=offset, dtype=dtype, name=name
        )
