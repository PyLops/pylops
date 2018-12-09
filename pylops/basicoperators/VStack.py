import numpy as np
from pylops import LinearOperator


class VStack(LinearOperator):
    r"""Vertical stacking.

    Stack a set of N linear operators vertically.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    Notes
    -----
    A vertical stack of N linear operators is created such as its application in
    forward mode leads to

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
        nops = np.zeros(len(ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            nops[iop] = oper.shape[0]
        self.nops = nops.sum()
        self.mops = ops[0].shape[1]
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.shape = (self.nops, self.mops)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.nnops[iop]:self.nnops[iop + 1]] = oper.matvec(x)
        return y

    def _rmatvec(self, x):
        y = np.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = y + oper.rmatvec(x[self.nnops[iop]:self.nnops[iop + 1]])
        return y
