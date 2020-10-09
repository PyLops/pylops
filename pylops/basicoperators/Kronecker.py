import numpy as np

from pylops import LinearOperator


class Kronecker(LinearOperator):
    r"""Kronecker operator.

    Perform Kronecker product of two operators. Note that the combined operator
    is never created explicitly, rather the product of this operator with the
    model vector is performed in forward mode, or the product of the adjoint of
    this operator and the data vector in adjoint mode.

    Parameters
    ----------
    Op1 : :obj:`pylops.LinearOperator`
        First operator
    Op2 : :obj:`pylops.LinearOperator`
        Second operator
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    The Kronecker product (denoted with :math:`\otimes`) is an operation
    on two operators :math:`\mathbf{Op_1}` and :math:`\mathbf{Op_2}` of
    sizes :math:`\lbrack n_1 \times m_1 \rbrack` and
    :math:`\lbrack n_2 \times m_2 \rbrack` respectively, resulting in a
    block matrix of size :math:`\lbrack n_1 n_2 \times m_1 m_2 \rbrack`.

    .. math::

        \mathbf{Op_1} \otimes \mathbf{Op_2} = \begin{bmatrix}
            Op_1^{1,1} \mathbf{Op_2} &  ... & Op_1^{1,m_1} \mathbf{Op_2}   \\
            ...                     &  ... & ... \\
            Op_1^{n_1,1} \mathbf{Op_2} &  ... & Op_1^{n_1,m_1} \mathbf{Op_2}
        \end{bmatrix}

    The application of the resulting matrix to a vector :math:`\mathbf{x}` of
    size :math:`\lbrack m_1 m_2 \times 1 \rbrack` is equivalent to the
    application of the second operator :math:`\mathbf{Op_2}` to the rows of
    a matrix of size :math:`\lbrack m_2 \times m_1 \rbrack` obtained by
    reshaping the input vector :math:`\mathbf{x}`, followed by the application
    of the first operator to the transposed matrix produced by the first
    operator. In adjoint mode the same procedure is followed but the adjoint of
    each operator is used.

    """
    def __init__(self, Op1, Op2, dtype='float64'):
        self.Op1 = Op1
        self.Op2 = Op2
        self.Op1H = self.Op1.H
        self.Op2H = self.Op2.H
        self.shape = (self.Op1.shape[0]*self.Op2.shape[0],
                      self.Op1.shape[1] * self.Op2.shape[1])
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = x.reshape(self.Op1.shape[1], self.Op2.shape[1])
        y = self.Op2.matmat(x.T).T
        y = self.Op1.matmat(y).ravel()
        return y

    def _rmatvec(self, x):
        x = x.reshape(self.Op1.shape[0], self.Op2.shape[0])
        y = self.Op2H.matmat(x.T).T
        y = self.Op1H.matmat(y).ravel()
        return y
