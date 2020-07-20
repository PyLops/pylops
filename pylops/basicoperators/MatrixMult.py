import numpy as np
from scipy.sparse.linalg import inv
from pylops import LinearOperator


class MatrixMult(LinearOperator):
    r"""Matrix multiplication.

    Simple wrapper to :py:func:`numpy.dot` and :py:func:`numpy.vdot` for
    an input matrix :math:`\mathbf{A}`.

    Parameters
    ----------
    A : :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrix
        Matrix.
    dims : :obj:`tuple`, optional
        Number of samples for each other dimension of model
        (model/data will be reshaped and ``A`` applied multiple times
        to each column of the model/data).
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    complex : :obj:`bool`
        Matrix has complex numbers (``True``) or not (``False``)

    """
    def __init__(self, A, dims=None, dtype='float64'):
        self.A = A
        if isinstance(A, np.ndarray):
            self.complex = np.iscomplexobj(A)
        else:
            self.complex = np.iscomplexobj(A.data)
        if dims is None:
            self.reshape = False
            self.shape = A.shape
            self.explicit = True
        else:
            if isinstance(dims, int):
                dims = (dims, )
            self.reshape = True
            self.dims = np.array(dims, dtype=np.int)
            self.reshapedims = \
                [np.insert([np.prod(self.dims)], 0, self.A.shape[1]),
                 np.insert([np.prod(self.dims)], 0, self.A.shape[0])]
            self.shape = (A.shape[0]*np.prod(self.dims),
                          A.shape[1]*np.prod(self.dims))
            self.explicit = False
        self.dtype = np.dtype(dtype)

    def _matvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.reshapedims[0])
        y = self.A.dot(x)
        if self.reshape:
            return y.ravel()
        else:
            return y

    def _rmatvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.reshapedims[1])
        if self.complex:
            y = (self.A.T.dot(x.conj())).conj()
        else:
            y = self.A.T.dot(x)

        if self.reshape:
            return y.ravel()
        else:
            return y

    def inv(self):
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        ----------
        Ainv : :obj:`numpy.ndarray`
            Inverse matrix.

        """
        if isinstance(self.A, np.ndarray):
            Ainv = np.linalg.inv(self.A)
        else:
            Ainv = inv(self.A)

        return Ainv
