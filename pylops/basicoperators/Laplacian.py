import numpy as np
from pylops.basicoperators import SecondDerivative


def Laplacian(dims, dirs=(0, 1), weights=(1, 1), sampling=(1, 1), dtype='float32'):
    r"""Laplacian.

    Apply second-order centered laplacian operator to a multi-dimensional array
    (at least 2 dimensions are required)

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    dirs : :obj:`tuple`, optional
        Directions along which laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if ``weights=[1,1]``)
    sampling : :obj:`tuple`, optional
        Sampling steps ``dx`` and ``dy`` for each direction
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    l2op : LinearOperator
        Laplacian linear operator

    Notes
    -----
    The Laplacian operator applies a second derivative along two directions of a
    multi-dimensional array.

    For simplicity, given a two dimensional array, the laplacin is:

    .. math::
        y[i, j] = (x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j]) / (dx*dy)

    """
    l2op = weights[0]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[0],
                                       sampling=sampling[0], dtype=dtype) + \
           weights[1]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[1],
                                       sampling=sampling[1], dtype=dtype)

    return l2op
