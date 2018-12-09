#from __future__ import division
import numpy as np
from pylops import LinearOperator


class MatrixMult(LinearOperator):
    r"""Matrix multiplication.

    Simple wrapper to :py:func:`numpy.dot` and :py:func:`numpy.vdot` for
    an input matrix :math:`\mathbf{A}`.

    Parameters
    ----------
    A : :obj:`numpy.ndarray`
        Matrix.
    dims : :obj:`tuple`, optional
        Number of samples for each other dimension of model
        (model will be reshaped and G applied multiple times to each column of the model/data).
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    """
    def __init__(self, A, dims=None, dtype='float32'):
        self.A = A
        if dims is None:
            self.reshape = False
            self.shape = A.shape
            self.explicit = True
        else:
            if isinstance(dims, int): dims = (dims, )
            self.reshape = True
            self.dims = np.array(dims, dtype=np.int)
            self.shape = (A.shape[0]*np.prod(self.dims),
                          A.shape[1]*np.prod(self.dims))
            self.explicit = False
        self.dtype = np.dtype(dtype)


    def _matvec(self, x):
        """ Apply forward matrix (y=G*x)
        :param np.ndarray x: vector

        :return: A*x
        :rtype np.ndarray
        """
        if self.reshape:
            x = np.reshape(x, np.insert([np.prod(self.dims)], 0, self.A.shape[1]))
        y = np.dot(self.A, x)
        if self.reshape:
            y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        """ Apply adjoint matrix (x=G'*y)
        :param np.ndarray y: vector

        :return: A'*y
        :rtype np.ndarray
        """
        if self.reshape:
            x = np.reshape(x, np.insert([np.prod(self.dims)], 0, self.A.shape[0]))
        y = np.dot(np.conj(self.A.T), x)
        if self.reshape:
            y = np.ndarray.flatten(y)
        return y

    def eigs(self):
        r"""Return eigenvalues of matrix :math:`\mathbf{A}`.

        Returns
        ----------
        eigenvalues : :obj:`numpy.ndarray`
            Matrix eigenvalues.

        """
        return -np.sort(-np.sqrt(np.linalg.eigvals(np.dot(self.A.T, self.A))))

    def inv(self):
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        ----------
        Ainv : :obj:`numpy.ndarray`
            Inverse matrix.

        """
        Ainv = np.linalg.inv(self.A)
        return Ainv
