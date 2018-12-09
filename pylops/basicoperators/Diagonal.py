import numpy as np
from pylops import LinearOperator


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with a vector :math:`\mathbf{d}`
    of the same size both in forward and adjoint mode (self-adjoint operator).

    Parameters
    ----------
    diag : :obj:`numpy.ndarray`
        Vector to be used for element-wise multiplication.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    Notes
    -----
    Element-wise multiplication between the model :math:`\mathbf{x}` and/or data :math:`\mathbf{y}`
    vectors and the array :math:`\mathbf{d}` can be expressed as

    .. math::

        y_i = d_i x_i  \quad \forall i=1,2,...,N

    This is equivalent to a matrix-vector multiplication with a matrix containing the vector
    :math:`\mathbf{d}` along its main diagonal. As the adjoint of a diagonal matrix is the
    diagonal matrix itself, the Diagonal is self.adjoint.

    """
    def __init__(self, diag, dtype=None):
        self.diag = diag.flatten()
        self.shape = (len(self.diag), len(self.diag))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return self.diag*x

    def _rmatvec(self, x):
        return self.diag*x

    def matrix(self):
        """Return diagonal matrix as dense :obj:`numpy.ndarray`

        Returns
        ----------
        densemat : :obj:`numpy.ndarray`
            Dense matrix.

        """
        densemat = np.diag(self.diag)
        return densemat
