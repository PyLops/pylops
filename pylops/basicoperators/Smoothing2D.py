import numpy as np
from pylops.signalprocessing import Convolve2D


def Smoothing2D(nsmooth, dims, nodir=None, dtype='float64'):
    r"""2D Smoothing.

    Apply smoothing to model (and data) along two directions of a
    multi-dimensional array depending on the choice of ``nodir``.

    Parameters
    ----------
    nsmooth : :obj:`tuple` or :obj:`list`
        Lenght of smoothing operatorin 1st and 2nd dimensions (must be odd)
    dims : :obj:`tuple`
        Number of samples for each dimension
    nodir : :obj:`int`, optional
        Direction along which smoothing is NOT applied (set to None for 2d
        arrays)
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
    lops.signalprocessing.Convolve2D : 2D convolution

    Notes
    -----
    The 2D Smoothing operator is a special type of convolutional operator that
    convolves the input model (or data) with a constant 2d filter of size
    :math:`n_{smooth, 1} \quad x \quad n_{smooth, 2}`:

    Its application to a two dimensional input signal is:

    .. math::
        y[i,j] = 1/(n_{smooth, 1}*n_{smooth, 2})
        \sum_{l=-(n_{smooth,1}-1)/2}^{(n_{smooth,1}-1)/2}
        \sum_{m=-(n_{smooth,2}-1)/2}^{(n_{smooth,2}-1)/2} x[l,m]

    Note that since the filter is symmetrical, the *Smoothing2D* operator is
    self-adjoint.

    """
    if isinstance(nsmooth, tuple):
        nsmooth = list(nsmooth)
    if nsmooth[0] % 2 == 0:
        nsmooth[0] += 1
    if nsmooth[1] % 2 == 0:
        nsmooth[1] += 1

    h = np.ones((nsmooth[0], nsmooth[1]))/float(nsmooth[0]*nsmooth[1])
    return Convolve2D(np.prod(np.array(dims)), h=h,
                      offset=[(nsmooth[0]-1)/2, (nsmooth[1]-1)/2],
                      dims=dims, nodir=nodir, dtype=dtype)
