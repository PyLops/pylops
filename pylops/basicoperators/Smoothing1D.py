import numpy as np
from pylops.signalprocessing import Convolve1D


def Smoothing1D(nsmooth, dims, dir=0, dtype='float64'):
    r"""1D Smoothing.

    Apply smoothing to model (and data) along a specific direction of a
    multi-dimensional array depending on the choice of ``dir``.

    Parameters
    ----------
    nsmooth : :obj:`int`
        Lenght of smoothing operator (must be odd)
    dims : :obj:`tuple` or :obj:`int`
        Number of samples for each dimension
    dir : :obj:`int`, optional
        Direction along which smoothing is applied
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    The Smoothing1D operator is a special type of convolutional operator that
    convolves the input model (or data) with a constant filter of size
    :math:`n_{smooth}`:

    .. math::
        \mathbf{f} = [ 1/n_{smooth}, 1/n_{smooth}, ..., 1/n_{smooth} ]

    When applied to the first direction:

    .. math::
        y[i,j,k] = 1/n_{smooth} \sum_{l=-(n_{smooth}-1)/2}^{(n_{smooth}-1)/2}
        x[l,j,k]

    Similarly when applied to the second direction:

    .. math::
        y[i,j,k] = 1/n_{smooth} \sum_{l=-(n_{smooth}-1)/2}^{(n_{smooth}-1)/2}
        x[i,l,k]

    and the third direction:

    .. math::
        y[i,j,k] = 1/n_{smooth} \sum_{l=-(n_{smooth}-1)/2}^{(n_{smooth}-1)/2}
        x[i,j,l]

    Note that since the filter is symmetrical, the *Smoothing1D* operator is
    self-adjoint.

    """
    if isinstance(dims, int):
        dims = (dims,)
    if nsmooth % 2 == 0:
        nsmooth += 1

    return Convolve1D(np.prod(np.array(dims)), dims=dims, dir=dir,
                      h=np.ones(nsmooth)/float(nsmooth), offset=(nsmooth-1)/2,
                      dtype=dtype)
