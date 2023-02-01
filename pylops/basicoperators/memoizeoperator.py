__all__ = ["MemoizeOperator"]

from typing import List, Tuple

import numpy as np

from pylops import LinearOperator
from pylops.optimization.base_linearoperator import BaseLinearOperator
from pylops.utils.typing import NDArray


class MemoizeOperator(BaseLinearOperator):
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

    def __init__(
        self,
        Op: LinearOperator,
        max_neval: int = 10,
    ) -> None:
        super().__init__(Op=Op)

        self.max_neval = max_neval
        self.store: List[Tuple[NDArray, NDArray]] = []  # Store a list of (x, y)
        self.neval = 0  # Number of evaluations of the operator

    def _matvec(self, x: NDArray) -> NDArray:
        for xstored, ystored in self.store:
            if np.allclose(xstored, x):
                return ystored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
        y = self.Op._matvec(x)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        return y

    def _rmatvec(self, y: NDArray) -> NDArray:
        for xstored, ystored in self.store:
            if np.allclose(ystored, y):
                return xstored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
        x = self.Op._rmatvec(y)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        return x
