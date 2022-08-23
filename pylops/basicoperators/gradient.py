__all__ = ["Gradient"]

from typing import List, Union

from pylops.basicoperators import FirstDerivative, VStack
from pylops.utils._internal import _value_or_sized_to_tuple


def Gradient(
    dims: Union[int, List[int]],
    sampling: int = 1,
    edge: bool = False,
    kind: str = "centered",
    dtype: str = "float64",
):
    r"""Gradient.

    Apply gradient operator to a multi-dimensional array.

    .. note:: At least 2 dimensions are required, use
      :py:func:`pylops.FirstDerivative` for 1d arrays.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``).
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    l2op : :obj:`pylops.LinearOperator`
        Gradient linear operator

    Notes
    -----
    The Gradient operator applies a first-order derivative to each dimension of
    a multi-dimensional array in forward mode.

    For simplicity, given a three dimensional array, the Gradient in forward
    mode using a centered stencil can be expressed as:

    .. math::
        \mathbf{g}_{i, j, k} =
            (f_{i+1, j, k} - f_{i-1, j, k}) / d_1 \mathbf{i_1} +
            (f_{i, j+1, k} - f_{i, j-1, k}) / d_2 \mathbf{i_2} +
            (f_{i, j, k+1} - f_{i, j, k-1}) / d_3 \mathbf{i_3}

    which is discretized as follows:

    .. math::
        \mathbf{g}  =
        \begin{bmatrix}
           \mathbf{df_1} \\
           \mathbf{df_2} \\
           \mathbf{df_3}
        \end{bmatrix}

    In adjoint mode, the adjoints of the first derivatives along different
    axes are instead summed together.

    """
    dims = _value_or_sized_to_tuple(dims)
    ndims = len(dims)
    sampling = _value_or_sized_to_tuple(sampling, repeat=ndims)

    gop = VStack(
        [
            FirstDerivative(
                dims,
                axis=iax,
                sampling=sampling[iax],
                edge=edge,
                kind=kind,
                dtype=dtype,
            )
            for iax in range(ndims)
        ]
    )
    gop.dims = dims
    gop.dimsd = (ndims, *gop.dims)
    gop.sampling = sampling
    gop.edge = edge
    gop.kind = kind
    return gop
