import logging

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Regression(LinearOperator):
    r"""Polynomial regression.

    Creates an operator that applies polynomial regression to a set of points.
    Values along the :math:`t`-axis must be provided while initializing the operator.
    The coefficients of the polynomial regression form the model vector to
    be provided in forward mode, while the values of the regression
    curve shall be provided in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`numpy.ndarray`
        Elements along the :math:`t`-axis.
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
        If ``taxis`` is not :obj:`numpy.ndarray`.

    See Also
    --------
    LinearRegression: Linear regression

    Notes
    -----
    The Regression operator solves the following problem:

    .. math::
        y_i = \sum_{n=0}^\text{order} x_n t_i^n  \qquad \forall i=0,1,\ldots,N-1

    where :math:`N` represents the number of points in ``taxis``. We can
    express this problem in a matrix form

    .. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

    where

    .. math::
        \mathbf{y}= [y_0, y_1,\ldots,y_{N-1}]^T,
        \qquad \mathbf{x}= [x_0, x_1,\ldots,x_\text{order}]^T

    and

    .. math::
        \mathbf{A}
        = \begin{bmatrix}
            1      & t_{0}  & t_{0}^2 & \ldots & t_{0}^\text{order}  \\
            1      & t_{1}  & t_{1}^2 & \ldots & t_{1}^\text{order}  \\
            \vdots & \vdots & \vdots  & \ddots & \vdots             \\
            1      & t_{N-1}  & t_{N-1}^2 & \ldots & t_{N-1}^\text{order}
        \end{bmatrix}_{N\times \text{order}+1}

    """

    def __init__(self, taxis, order, dtype="float64"):
        ncp = get_array_module(taxis)
        if not isinstance(taxis, ncp.ndarray):
            logging.error("t must be numpy.ndarray...")
            raise TypeError("t must be numpy.ndarray...")
        else:
            self.taxis = taxis
        self.order = order
        self.shape = (len(self.taxis), self.order + 1)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros_like(self.taxis)
        for i in range(self.order + 1):
            y += x[i] * self.taxis ** i
        return y

    def _rmatvec(self, x):
        ncp = get_array_module(x)

        return ncp.vstack([ncp.dot(self.taxis ** i, x) for i in range(self.order + 1)])

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
