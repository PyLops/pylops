import numpy as np
from pylops.basicoperators import FirstDerivative, VStack


def Gradient(dims, sampling=1, edge=False, dtype='float64', kind='centered'):
    r"""Gradient.

    Apply gradient operator to a multi-dimensional
    array (at least 2 dimensions are required).

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).

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
    ndims = len(dims)
    if isinstance(sampling, (int, float)):
        sampling = [sampling] * ndims

    gop = VStack([FirstDerivative(np.prod(dims), dims=dims, dir=idir,
                                  sampling=sampling[idir],
                                  edge=edge, kind=kind, dtype=dtype)
                  for idir in range(ndims)])
    return gop
