__all__ = ["Kronecker"]

import numpy as np

from pylops import LinearOperator
from pylops.utils.typing import DTypeLike, NDArray


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
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

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
    on two operators :math:`\mathbf{Op}_1` and :math:`\mathbf{Op}_2` of
    sizes :math:`\lbrack n_1 \times m_1 \rbrack` and
    :math:`\lbrack n_2 \times m_2 \rbrack` respectively, resulting in a
    block matrix of size :math:`\lbrack n_1 n_2 \times m_1 m_2 \rbrack`.

    .. math::

        \mathbf{Op}_1 \otimes \mathbf{Op}_2 = \begin{bmatrix}
            \text{Op}_1^{1,1} \mathbf{Op}_2   &  \ldots & \text{Op}_1^{1,m_1} \mathbf{Op}_2   \\
            \vdots                            &  \ddots & \vdots \\
            \text{Op}_1^{n_1,1} \mathbf{Op}_2 &  \ldots & \text{Op}_1^{n_1,m_1} \mathbf{Op}_2
        \end{bmatrix}

    The application of the resulting matrix to a vector :math:`\mathbf{x}` of
    size :math:`\lbrack m_1 m_2 \times 1 \rbrack` is equivalent to the
    application of the second operator :math:`\mathbf{Op}_2` to the rows of
    a matrix of size :math:`\lbrack m_2 \times m_1 \rbrack` obtained by
    reshaping the input vector :math:`\mathbf{x}`, followed by the application
    of the first operator to the transposed matrix produced by the first
    operator. In adjoint mode the same procedure is followed but the adjoint of
    each operator is used.

    """

    def __init__(
        self,
        Op1: LinearOperator,
        Op2: LinearOperator,
        dtype: DTypeLike = "float64",
        name: str = "K",
    ) -> None:
        self.Op1 = Op1
        self.Op2 = Op2
        self.Op1H = self.Op1.H
        self.Op2H = self.Op2.H
        shape = (
            self.Op1.shape[0] * self.Op2.shape[0],
            self.Op1.shape[1] * self.Op2.shape[1],
        )
        super().__init__(dtype=np.dtype(dtype), shape=shape, name=name)

    def _matvec(self, x: NDArray) -> NDArray:
        x = x.reshape(self.Op1.shape[1], self.Op2.shape[1])
        y = self.Op2.matmat(x.T).T
        y = self.Op1.matmat(y).ravel()
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        x = x.reshape(self.Op1.shape[0], self.Op2.shape[0])
        y = self.Op2H.matmat(x.T).T
        y = self.Op1H.matmat(y).ravel()
        return y
