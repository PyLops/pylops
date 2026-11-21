__all__ = ["Smoothing2D"]


import numpy as np

from pylops.signalprocessing import Convolve2D
from pylops.utils.typing import DTypeLike, InputDimsLike


class Smoothing2D(Convolve2D):
    r"""2D Smoothing.

    Apply smoothing to model (and data) along two ``axes`` of a
    multi-dimensional array.

    Parameters
    ----------
    nsmooth : :obj:`tuple` or :obj:`list`
        Length of smoothing operator in 1st and 2nd dimensions
        (must be odd, if even it will be increased by 1).
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`tuple` or :obj:`list`, optional
        .. versionadded:: 2.0.0

        Axes along which model (and data) are smoothed. If ``None``, use the
        last two axes. Must contain two unique axes. Note that the i-th entry
        of ``nsmooth`` is applied along ``axes[i]``; as such, ``axes`` does
        not need to be sorted: for example, with ``nsmooth=(3, 5)`` and
        ``axes=(2, 0)`` a window of length 3 is used along axis 2 and of
        length 5 along axis 0.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    nh : :obj:`tuple`
        Length of the filter
    convolve : :obj:`callable`
        Convolution function
    correlate : :obj:`callable`
        Correlation function
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    See Also
    --------
    pylops.signalprocessing.Convolve2D : 2D convolution

    Notes
    -----
    The 2D Smoothing operator is a special type of convolutional operator that
    convolves the input model (or data) with a constant 2d filter of size
    :math:`n_{\text{smooth}, 1} \times n_{\text{smooth}, 2}`:

    Its application to a two dimensional input signal is:

    .. math::
        y[i,j] = 1/(n_{\text{smooth}, 1}\cdot n_{\text{smooth}, 2})
        \sum_{l=-(n_{\text{smooth},1}-1)/2}^{(n_{\text{smooth},1}-1)/2}
        \sum_{m=-(n_{\text{smooth},2}-1)/2}^{(n_{\text{smooth},2}-1)/2} x[l,m]

    Note that since the filter is symmetrical, the *Smoothing2D* operator is
    self-adjoint.

    """

    def __init__(
        self,
        nsmooth: InputDimsLike,
        dims: int | InputDimsLike,
        axes: InputDimsLike | None = None,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ):
        nsmooth = list(nsmooth)
        if nsmooth[0] % 2 == 0:
            nsmooth[0] += 1
        if nsmooth[1] % 2 == 0:
            nsmooth[1] += 1
        h = np.ones((nsmooth[0], nsmooth[1]), dtype=dtype) / float(
            nsmooth[0] * nsmooth[1]
        )
        offset = [(nsmooth[0] - 1) // 2, (nsmooth[1] - 1) // 2]
        super().__init__(
            dims=dims, h=h, offset=offset, axes=axes, dtype=dtype, name=name
        )
