from pylops.basicoperators import HStack, VStack


def _Block(
    ops, dtype=None, _HStack=HStack, _VStack=VStack, args_HStack={}, args_VStack={}
):
    """Block operator.

    Used to be able to provide operators from different libraries to
    Block.
    """
    hblocks = [_HStack(hblock, dtype=dtype, **args_HStack) for hblock in ops]
    return _VStack(hblocks, dtype=dtype, **args_VStack)


def Block(ops, nproc=1, dtype=None):
    r"""Block operator.

    Create a block operator from N lists of M linear operators each.

    Parameters
    ----------
    ops : :obj:`list`
        List of lists of operators to be combined in block fashion.
        Alternatively, :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices
        can be passed in place of one or more operators.
    nproc : :obj:`int`, optional
        Number of processes used to evaluate the N operators in parallel using
        ``multiprocessing``. If ``nproc=1``, work in serial mode.
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
            \mathbf{L}_{1,1}  & \mathbf{L}_{1,2} &  \ldots & \mathbf{L}_{1,M}  \\
            \mathbf{L}_{2,1}  & \mathbf{L}_{2,2} &  \ldots & \mathbf{L}_{2,M}  \\
            \vdots            & \vdots           &  \ddots & \vdots            \\
            \mathbf{L}_{N,1}  & \mathbf{L}_{N,2} &  \ldots & \mathbf{L}_{N,M}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots          \\
            \mathbf{x}_{M}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L}_{1,1} \mathbf{x}_{1} + \mathbf{L}_{1,2} \mathbf{x}_{2} +
            \mathbf{L}_{1,M} \mathbf{x}_{M} \\
            \mathbf{L}_{2,1} \mathbf{x}_{1} + \mathbf{L}_{2,2} \mathbf{x}_{2} +
            \mathbf{L}_{2,M} \mathbf{x}_{M} \\
            \vdots     \\
            \mathbf{L}_{N,1} \mathbf{x}_{1} + \mathbf{L}_{N,2} \mathbf{x}_{2} +
            \mathbf{L}_{N,M} \mathbf{x}_{M}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1,1}^H  & \mathbf{L}_{2,1}^H & \ldots & \mathbf{L}_{N,1}^H  \\
            \mathbf{L}_{1,2}^H  & \mathbf{L}_{2,2}^H & \ldots & \mathbf{L}_{N,2}^H  \\
            \vdots              & \vdots             & \ddots & \vdots \\
            \mathbf{L}_{1,M}^H  & \mathbf{L}_{2,M}^H & \ldots & \mathbf{L}_{N,M}^H
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L}_{1,1}^H \mathbf{y}_{1} +
            \mathbf{L}_{2,1}^H \mathbf{y}_{2} +
            \mathbf{L}_{N,1}^H \mathbf{y}_{N} \\
            \mathbf{L}_{1,2}^H \mathbf{y}_{1} +
            \mathbf{L}_{2,2}^H \mathbf{y}_{2} +
            \mathbf{L}_{N,2}^H \mathbf{y}_{N} \\
            \vdots     \\
            \mathbf{L}_{1,M}^H \mathbf{y}_{1} +
            \mathbf{L}_{2,M}^H \mathbf{y}_{2} +
            \mathbf{L}_{N,M}^H \mathbf{y}_{N}
        \end{bmatrix}

    """
    return _Block(ops, dtype=dtype, args_VStack={"nproc": nproc})
