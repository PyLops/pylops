import numpy as np

from pylops import LinearOperator


class MemoizeOperator(LinearOperator):
    r"""Memoize Operator.

    This operator can be used to wrap any PyLops operator and add a memoize
    functionality
    and stores the last ``max_neval`` model/data
    vector pairs

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        PyLops linear operator
    max_neval : :obj:`int`, optional
        Maximum number of previous evaluations stored,
        use ``np.inf`` for infinite memory

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape :math:`[n \times m]`
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    """

    def __init__(self, Op, max_neval=10):
        self.Op = Op
        self.shape = Op.shape
        self.dtype = np.dtype(Op.dtype)
        self.explicit = False

        self.max_neval = max_neval
        self.store = []  # Store a list of Tuples (x, y)
        self.neval = 0  # Number of evaluations of the operator

    def _matvec(self, x):
        for xstored, ystored in self.store:
            if np.allclose(xstored, x):
                return ystored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
        y = self.Op._matvec(x)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        return y

    def _rmatvec(self, y):
        for xstored, ystored in self.store:
            if np.allclose(ystored, y):
                return xstored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
        x = self.Op._rmatvec(y)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        return x
