from pylops.basicoperators import HStack, VStack

def Block(ops, dtype=None):
    r"""Block operator.

    Create a block operator from N lists of M linear operators each.

    Parameters
    ----------
    ops : :obj:`list`
        List of lists of operators to be combined in block fashion
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
    In mathematics, a block or a partitioned matrix is a matrix that is
    interpreted as being broken into sections called blocks or submatrices.
    Similarly a block operator is composed of N sets of M linear operators
    each such that its application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_{1,1}}  & \mathbf{L_{1,2}} &  ... & \mathbf{L_{1,M}}  \\
            \mathbf{L_{2,1}}  & \mathbf{L_{2,2}} &  ... & \mathbf{L_{2,M}}  \\
            ...               & ...              &  ... & ...               \\
            \mathbf{L_{N,1}}  & \mathbf{L_{N,2}} &  ... & \mathbf{L_{N,M}}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            ...     \\
            \mathbf{x}_{M}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_{1,1}} \mathbf{x}_{1} + \mathbf{L_{1,2}} \mathbf{x}_{2} +
            \mathbf{L_{1,M}} \mathbf{x}_{M} \\
            \mathbf{L_{2,1}} \mathbf{x}_{1} + \mathbf{L_{2,2}} \mathbf{x}_{2} +
            \mathbf{L_{2,M}} \mathbf{x}_{M} \\
            ...     \\
            \mathbf{L_{N,1}} \mathbf{x}_{1} + \mathbf{L_{N,2}} \mathbf{x}_{2} +
            \mathbf{L_{N,M}} \mathbf{x}_{M} \\
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_{1,1}}^H  & \mathbf{L_{2,1}}^H &  ... &
            \mathbf{L_{N,1}}^H  \\
            \mathbf{L_{1,2}}^H  & \mathbf{L_{2,2}}^H &  ... &
            \mathbf{L_{N,2}}^H  \\
            ...                 & ...                &  ... & ... \\
            \mathbf{L_{1,M}}^H  & \mathbf{L_{2,M}}^H &  ... &
            \mathbf{L_{N,M}}^H  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            ...     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_{1,1}}^H \mathbf{y}_{1} +
            \mathbf{L_{2,1}}^H \mathbf{y}_{2} +
            \mathbf{L_{N,1}}^H \mathbf{y}_{N} \\
            \mathbf{L_{1,2}}^H \mathbf{y}_{1} +
            \mathbf{L_{2,2}}^H \mathbf{y}_{2} +
            \mathbf{L_{N,2}}^H \mathbf{y}_{N} \\
            ...     \\
            \mathbf{L_{1,M}}^H \mathbf{y}_{1} +
            \mathbf{L_{2,M}}^H \mathbf{y}_{2} +
            \mathbf{L_{N,M}}^H \mathbf{y}_{N} \\
        \end{bmatrix}

    """
    hblocks = [HStack(hblock, dtype=dtype) for hblock in ops]
    return VStack(hblocks, dtype=dtype)
