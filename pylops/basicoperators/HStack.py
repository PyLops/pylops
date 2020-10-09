import numpy as np
from scipy.sparse.linalg.interface import _get_dtype
from scipy.sparse.linalg.interface import LinearOperator as spLinearOperator
from pylops import LinearOperator
from pylops.basicoperators import MatrixMult


class HStack(LinearOperator):
    r"""Horizontal stacking.

    Stack a set of N linear operators horizontally.

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
        If ``ops`` have different number of columns

    Notes
    -----
    An horizontal stack of N linear operators is created such as its
    application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1} & \mathbf{L}_{2} & ... & \mathbf{L}_{N}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            ...     \\
            \mathbf{x}_{N}
        \end{bmatrix} =
        \mathbf{L}_{1} \mathbf{x}_1 + \mathbf{L}_{2} \mathbf{x}_2 +
        ... + \mathbf{L}_{N} \mathbf{x}_N

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}^H  \\
            \mathbf{L}_{2}^H  \\
            ...     \\
            \mathbf{L}_{N}^H
        \end{bmatrix}
        \mathbf{y} =
        \begin{bmatrix}
            \mathbf{L}_{1}^H \mathbf{y}  \\
            \mathbf{L}_{2}^H \mathbf{y}  \\
            ...     \\
            \mathbf{L}_{N}^H \mathbf{y}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            ...     \\
            \mathbf{x}_{N}
        \end{bmatrix}

    """
    def __init__(self, ops, dtype=None):
        self.ops = ops
        mops = np.zeros(len(ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            mops[iop] = self.ops[iop].shape[1]
        self.mops = mops.sum()
        nops = [oper.shape[0] for oper in self.ops]
        if len(set(nops)) > 1:
            raise ValueError('operators have different number of rows')
        self.nops = nops[0]
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(self.ops)
        else:
            self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y += oper.matvec(x[self.mmops[iop]:self.mmops[iop + 1]]).squeeze()
        return y

    def _rmatvec(self, x):
        y = np.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.mmops[iop]:self.mmops[iop + 1]] = oper.rmatvec(x).squeeze()
        return y
