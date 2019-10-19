import numpy as np
from scipy.sparse.linalg.interface import _get_dtype
from scipy.sparse.linalg.interface import LinearOperator as spLinearOperator
from pylops import LinearOperator
from pylops.basicoperators import MatrixMult


class BlockDiag(LinearOperator):
    r"""Block-diagonal operator.

    Create a block-diagonal operator from N linear operators.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked. Alternatively,
        :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
        in place of one or more operators.
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
    A block-diagonal operator composed of N linear operators is created such
    as its application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_1}  & \mathbf{0}   &  ... &  \mathbf{0}  \\
            \mathbf{0}    & \mathbf{L_2} &  ... &  \mathbf{0}  \\
            ...           & ...          &  ... &  ...         \\
            \mathbf{0}    & \mathbf{0}   &  ... &  \mathbf{L_N}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            ...     \\
            \mathbf{x}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_1} \mathbf{x}_{1}  \\
            \mathbf{L_2} \mathbf{x}_{2}  \\
            ...     \\
            \mathbf{L_N} \mathbf{x}_{N}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_1}^H  \quad \mathbf{0}    \quad ... \quad  \mathbf{0}  \\
            \mathbf{0}    \quad \mathbf{L_2}^H  \quad ... \quad  \mathbf{0}  \\
            ...           \quad ...             \quad ... \quad  ...         \\
            \mathbf{0}    \quad \mathbf{0}      \quad ... \quad  \mathbf{L_N}^H
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            ...     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_1}^H \mathbf{y}_{1}  \\
            \mathbf{L_2}^H \mathbf{y}_{2}  \\
            ...     \\
            \mathbf{L_N}^H \mathbf{y}_{N}
        \end{bmatrix}

    """
    def __init__(self, ops, dtype=None):
        self.ops = ops
        mops = np.zeros(len(ops), dtype=np.int)
        nops = np.zeros(len(ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = self.ops[iop].shape[0]
            mops[iop] = self.ops[iop].shape[1]
        self.nops = nops.sum()
        self.mops = mops.sum()
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(ops)
        else:
            self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.nnops[iop]:self.nnops[iop + 1]] = \
                oper.matvec(x[self.mmops[iop]:self.mmops[iop + 1]]).squeeze()
        return y

    def _rmatvec(self, x):
        y = np.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.mmops[iop]:self.mmops[iop + 1]] = \
                oper.rmatvec(x[self.nnops[iop]:self.nnops[iop + 1]]).squeeze()
        return y
