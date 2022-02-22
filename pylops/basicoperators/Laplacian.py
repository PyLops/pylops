import warnings

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops.basicoperators import SecondDerivative
from pylops.LinearOperator import aslinearoperator


def Laplacian(
    dims,
    axes=(-2, -1),
    dirs=None,
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
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which the Laplacian is applied.
    dirs : :obj:`int`, optional

        .. deprecated:: 2.0.0
            Use ``axes`` instead. Note that the default for ``axes`` is (-2, -1)
            instead of (0, 1) which was the default for ``dirs``.

    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=(1, 1)``)
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
        If ``axes``. ``weights``, and ``sampling`` do not have the same size

    Notes
    -----
    The Laplacian operator applies a second derivative along two directions of
    a multi-dimensional array.

    For simplicity, given a two dimensional array, the Laplacian is:

    .. math::
        y[i, j] = (x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j])
                  / (\Delta x \Delta y)

    """
    if dirs is not None:
        warnings.warn(
            "dirs will be deprecated in version 2.0.0, use axes instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        axes = dirs
    axes = tuple(normalize_axis_index(ax, len(dims)) for ax in axes)
    if not (len(axes) == len(weights) == len(sampling)):
        raise ValueError("axes, weights, and sampling have different size")

    l2op = weights[0] * SecondDerivative(
        dims, axis=axes[0], sampling=sampling[0], edge=edge, kind=kind, dtype=dtype
    )

    for ax, samp, weight in zip(axes[1:], sampling[1:], weights[1:]):
        l2op += weight * SecondDerivative(
            dims, axis=ax, sampling=samp, edge=edge, dtype=dtype
        )

    return aslinearoperator(l2op)
