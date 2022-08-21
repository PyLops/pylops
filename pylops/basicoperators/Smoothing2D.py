from typing import Tuple, Union

import numpy as np

from pylops.signalprocessing import Convolve2D


def Smoothing2D(
    nsmooth: int,
    dims: Union[int, Tuple[int]],
    axes: Tuple[int] = (-2, -1),
    dtype: str = "float64",
) -> None:
    r"""2D Smoothing.

    Apply smoothing to model (and data) along two ``axes`` of a
    multi-dimensional array.

    Parameters
    ----------
    nsmooth : :obj:`tuple` or :obj:`list`
        Lenght of smoothing operator in 1st and 2nd dimensions (must be odd)
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which model (and data) are smoothed.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

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
    if isinstance(nsmooth, tuple):
        nsmooth = list(nsmooth)
    if nsmooth[0] % 2 == 0:
        nsmooth[0] += 1
    if nsmooth[1] % 2 == 0:
        nsmooth[1] += 1
    h = np.ones((nsmooth[0], nsmooth[1])) / float(nsmooth[0] * nsmooth[1])
    offset = [(nsmooth[0] - 1) / 2, (nsmooth[1] - 1) / 2]
    return Convolve2D(dims, h=h, offset=offset, axes=axes, dtype=dtype)
