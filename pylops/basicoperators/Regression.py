import logging

import numpy as np
from pylops import LinearOperator

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

class Regression(LinearOperator):
    r"""Polynomial regression.

    Creates an operator that applies polynomial regression to a set of points.
    Values along the t-axis must be provided while initializing the operator.
    The coefficients of the polynomial regression form the model vector to
    be provided in forward mode, while the values of the regression
    curve shall be provided in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`numpy.ndarray`
        Elements along the t-axis.
    order : :obj:`int`
        Order of the regressed polynomial.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    See Also
    --------
    LinearRegression: Linear regression

    Notes
    -----
    The Regression operator solves the following problem:

    .. math::
        y_i = \sum_{n=0}^{order} x_n t_i^n  \qquad \forall i=1,2,...,N

    where :math:`N` represents the order of the chosen polynomial. We can
    express this problem in a matrix form

    .. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

    where

    .. math::
        \mathbf{y}= [y_1, y_2,...,y_N]^T,
        \qquad \mathbf{x}= [x_0, x_1,...,x_{order}]^T

    and

    .. math::
        \mathbf{A}
        = \begin{bmatrix}
            1  & t_{1} & t_{1}^2 & .. & t_{1}^{order}  \\
            1  & t_{2} & t_{2}^2 & .. & t_{1}^{order}  \\
            .. & ..    & ..      & .. & ..             \\
            1  & t_{N} & t_{N}^2 & .. & t_{N}^{order}  \\
        \end{bmatrix}

    """
    def __init__(self, taxis, order, dtype='float64'):
        if not isinstance(taxis, np.ndarray):
            logging.error('t must be numpy.ndarray...')
            raise TypeError('t must be numpy.ndarray...')
        else:
            self.taxis = taxis
        self.order = order
        self.shape = (len(self.taxis), self.order+1)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = np.zeros_like(self.taxis)
        for i in range(self.order+1):
            y += x[i]*self.taxis**i
        return y

    def _rmatvec(self, x):
        return np.vstack([np.dot(self.taxis**i, x)
                          for i in range(self.order+1)])

    def apply(self, t, x):
        """Return values along y-axis given certain ``t`` location(s) along
        t-axis and regression coefficients ``x``

        Parameters
        ----------
        taxis : :obj:`numpy.ndarray`
            Elements along the t-axis.
        x : :obj:`numpy.ndarray`
            Regression coefficients
        dtype : :obj:`str`, optional
        Returns
        ----------
        y : :obj:`numpy.ndarray`
            Values along y-axis

        """
        torig = self.taxis.copy()
        self.taxis = t
        y = self._matvec(x)
        self.taxis = torig
        return y
