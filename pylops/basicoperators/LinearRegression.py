import logging

import numpy as np
from pylops import LinearOperator

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

class LinearRegression(LinearOperator):
    r"""Linear regression operator.

    Creates an operator that applies linear regression to a set of points.
    Values along the t-axis (or x-axis) has to be provided while initializing the operator.
    The intercept and gradient are the model coefficients to be provided in forward mode,
    while the values along y-axis regression line y values are provided in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`numpy.ndarray`
        Elements along the t-axis.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    Notes
    -----
    The Linear regression operators solves the following problem:

    .. math::
        y_i = x_0 + x_1 t_i  \qquad \forall i=1,2,...,N

    We can express this problem in a matrix form

    .. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

    where

    .. math::
        \mathbf{y}= [y_1, y_2,...,y_N]^T, \qquad \mathbf{x}= [x_0, x_1]^T

    and

    .. math::
        \mathbf{A}
        = \begin{bmatrix}
            1       & t_{1}  \\
            1       & t_{2}  \\
            ..      & ..     \\
            1       & t_{N}
        \end{bmatrix}

    """
    def __init__(self, taxis, dtype='float32'):
        if not isinstance(taxis, np.ndarray):
            logging.error('t must be numpy.ndarray...')
            raise TypeError('t must be numpy.ndarray...')
        else:
            self.taxis = taxis
        self.shape = (len(self.taxis), 2)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return x[0]+x[1]*self.taxis

    def _rmatvec(self, x):
        return np.vstack((np.sum(x), np.dot(self.taxis, x)))
