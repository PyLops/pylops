from __future__ import division

from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from scipy.sparse.linalg import lsqr


class LinearOperator(spLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is a wrapper of the :py:class:`scipy.sparse.linalg.LinearOperator` class,
    which contains additional overloading to standard operators such as ``__div__``.

    """
    def __init__(self, explicit=False):
        super(LinearOperator, self).__init__()
        self.explicit = explicit

    def div(self, y):
        r"""Solve the linear problem :math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

        Overloading of operator ``/`` to improve expressivity of `Pylops`
        when solving inverse problems.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data

        Returns
        -------
        xest : :obj:`np.ndarray`
            Estimated model

        """
        xest = self.__truediv__(y)
        return xest

    def __truediv__(self, y):
        if self.explicit is True:
            xest = solve(self.A, y) if self.A.shape[0] == self.A.shape[1] else lstsq(self.A, y)[0]
        else:
            xest = lsqr(self, y)[0]

        return xest
