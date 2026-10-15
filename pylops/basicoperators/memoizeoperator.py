__all__ = ["MemoizeOperator"]


import numpy as np

from pylops import LinearOperator
from pylops.utils.typing import NDArray


class MemoizeOperator(LinearOperator):
    r"""Memoize Operator.

    This operator can be used to wrap any PyLops operator and add a memoize
    functionality
    and stores the last ``max_neval`` model/data
    vector pairs. Forward and adjoint evaluations are cached separately,
    sharing the same ``max_neval`` limit. An adjoint is not assumed to be
    the inverse of a forward evaluation.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        PyLops linear operator
    max_neval : :obj:`int`, optional
        Maximum number of previous evaluations stored,
        use ``np.inf`` for infinite memory

    Attributes
    ----------
    store : :obj:`list`
        List of stored (x, y) pairs.
    neval : :obj:`int`
        Number of evaluations of the operator.
    shape : :obj:`tuple`
        Operator shape. :math:`[n \times m]`.

    """

    def __init__(
        self,
        Op: LinearOperator,
        max_neval: int = 10,
    ) -> None:
        super().__init__(Op=Op)

        self.max_neval = max_neval
        self.store: list[tuple[NDArray, NDArray]] = []  # Store a list of (x, y)
        self._store_is_adjoint: list[bool] = []
        self.neval = 0  # Number of evaluations of the operator

    def _matvec(self, x: NDArray) -> NDArray:
        for (xstored, ystored), is_adjoint in zip(
            self.store, self._store_is_adjoint, strict=True
        ):
            if not is_adjoint and np.allclose(xstored, x):
                return ystored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
            del self._store_is_adjoint[0]
        y = self.Op._matvec(x)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        self._store_is_adjoint.append(False)
        return y

    def _rmatvec(self, y: NDArray) -> NDArray:
        for (xstored, ystored), is_adjoint in zip(
            self.store, self._store_is_adjoint, strict=True
        ):
            if is_adjoint and np.allclose(ystored, y):
                return xstored
        if len(self.store) + 1 > self.max_neval:
            del self.store[0]  # Delete oldest
            del self._store_is_adjoint[0]
        x = self.Op._rmatvec(y)
        self.neval += 1
        self.store.append((x.copy(), y.copy()))
        self._store_is_adjoint.append(True)
        return x
