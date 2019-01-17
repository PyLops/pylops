import logging
from pylops.basicoperators import Regression

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

def LinearRegression(taxis, dtype='float64'):
    r"""Linear regression.

    Creates an operator that applies linear regression to a set of points.
    Values along the t-axis  must be provided while initializing the operator.
    Intercept and gradient form the model vector to be provided in forward
    mode, while the values of the regression line curve shall be provided
    in adjoint mode.

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
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    See Also
    --------
    Regression: Polynomial regression

    Notes
    -----
    The LinearRegression operator solves the following problem:

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

    Note that this is a particular case of the :py:class:`pylops.Regression`
    operator and it is in fact just a lazy call of that operator with
    ``order=1``.
    """
    return Regression(taxis, order=1, dtype=dtype)
