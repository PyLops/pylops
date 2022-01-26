import logging

from pylops.basicoperators import Regression

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def LinearRegression(taxis, dtype="float64"):
    r"""Linear regression.

    Creates an operator that applies linear regression to a set of points.
    Values along the :math:`t`-axis  must be provided while initializing the operator.
    Intercept and gradient form the model vector to be provided in forward
    mode, while the values of the regression line curve shall be provided
    in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`numpy.ndarray`
        Elements along the :math:`t`-axis.
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
    Regression: Polynomial regression

    Notes
    -----
    The LinearRegression operator solves the following problem:

    .. math::
        y_i = x_0 + x_1 t_i  \qquad \forall i=0,1,\ldots,N-1

    We can express this problem in a matrix form

    .. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

    where

    .. math::
        \mathbf{y}= [y_0, y_1,\ldots,y_{N-1}]^T, \qquad \mathbf{x}= [x_0, x_1]^T

    and

    .. math::
        \mathbf{A}
        = \begin{bmatrix}
            1       & t_{0}  \\
            1       & t_{1}  \\
            \vdots      & \vdots     \\
            1       & t_{N-1}
        \end{bmatrix}

    Note that this is a particular case of the :py:class:`pylops.Regression`
    operator and it is in fact just a lazy call of that operator with
    ``order=1``.
    """
    return Regression(taxis, order=1, dtype=dtype)
