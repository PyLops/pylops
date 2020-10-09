import numpy as np
from scipy.sparse.linalg.interface import _get_dtype
from scipy.sparse.linalg.interface import LinearOperator as spLinearOperator
from pylops import LinearOperator
from pylops.basicoperators import MatrixMult


class VStack(LinearOperator):
    r"""Vertical stacking.

    Stack a set of N linear operators vertically.

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

    Raises
    ------
    ValueError
        If ``ops`` have different number of rows

    Notes
    -----
    A vertical stack of N linear operators is created such as its application
    in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}  \\
            \mathbf{L}_{2}  \\
            ...     \\
            \mathbf{L}_{N}
        \end{bmatrix}
        \mathbf{x} =
        \begin{bmatrix}
            \mathbf{L}_{1} \mathbf{x}  \\
            \mathbf{L}_{2} \mathbf{x}  \\
            ...     \\
            \mathbf{L}_{N} \mathbf{x}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            ...     \\
            \mathbf{y}_{N}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}^H & \mathbf{L}_{2}^H & ... & \mathbf{L}_{N}^H
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            ...     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \mathbf{L}_{1}^H \mathbf{y}_1 + \mathbf{L}_{2}^H \mathbf{y}_2 +
        ... + \mathbf{L}_{N}^H \mathbf{y}_N

    """
    def __init__(self, ops, dtype=None):
        self.ops = ops
        nops = np.zeros(len(self.ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = self.ops[iop].shape[0]
        self.nops = nops.sum()
        mops = [oper.shape[1] for oper in self.ops]
        if len(set(mops)) > 1:
            raise ValueError('operators have different number of columns')
        self.mops = mops[0]
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(self.ops)
        else:
            self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.nnops[iop]:self.nnops[iop + 1]] = oper.matvec(x).squeeze()
        return y

    def _rmatvec(self, x):
        y = np.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y += oper.rmatvec(x[self.nnops[iop]:self.nnops[iop + 1]]).squeeze()
        return y
