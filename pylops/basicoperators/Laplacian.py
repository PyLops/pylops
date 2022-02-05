import numpy as np

from pylops.basicoperators import SecondDerivative
from pylops.LinearOperator import aslinearoperator


def Laplacian(
    dims,
    dirs=(0, 1),
    weights=(1, 1),
    sampling=(1, 1),
    edge=False,
    dtype="float64",
    kind="centered",
):
    r"""Laplacian.

    Apply second-order centered Laplacian operator to a multi-dimensional array.

    .. note:: At least 2 dimensions are required, use
      :py:func:`pylops.SecondDerivative` for 1d arrays.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    dirs : :obj:`tuple`, optional
        Directions along which laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=[1,1]``)
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``) for centered derivative
    dtype : :obj:`str`, optional
        Type of elements in input array.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``)

    Returns
    -------
    l2op : :obj:`pylops.LinearOperator`
        Laplacian linear operator

    Raises
    ------
    ValueError
        If ``dirs``. ``weights``, and ``sampling`` do not have the same size

    Notes
    -----
    The Laplacian operator applies a second derivative along two directions of
    a multi-dimensional array.

    For simplicity, given a two dimensional array, the Laplacian is:

    .. math::
        y[i, j] = (x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j])
                  / (\Delta x \Delta y)

    """
    if not (len(dirs) == len(weights) == len(sampling)):
        raise ValueError("dirs, weights, and sampling have different size")

    l2op = weights[0] * SecondDerivative(
        np.prod(dims),
        dims=dims,
        dir=dirs[0],
        sampling=sampling[0],
        edge=edge,
        kind=kind,
        dtype=dtype,
    )
    
    for dir, samp, weight in zip(dirs[1:], sampling[1:], weights[1:]):
        l2op += weight * SecondDerivative(
            np.prod(dims),
            dims=dims,
            dir=dir,
            sampling=samp,
            edge=edge,
            dtype=dtype,
        )
    
    return aslinearoperator(l2op)
