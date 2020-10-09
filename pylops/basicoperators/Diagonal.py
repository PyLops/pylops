import numpy as np
from pylops import LinearOperator


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with the vector
    ``diag`` in forward and with its complex conjugate in adjoint mode.

    This operator can also broadcast; in this case the input vector is
    reshaped into its dimensions ``dims`` and the element-wise multiplication
    with ``diag`` is perfomed on the direction ``dir``. Note that the
    vector ``diag`` will need to have size equal to ``dims[dir]``.

    Parameters
    ----------
    diag : :obj:`numpy.ndarray`
        Vector to be used for element-wise multiplication.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which multiplication is applied.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Element-wise multiplication between the model :math:`\mathbf{x}` and/or
    data :math:`\mathbf{y}` vectors and the array :math:`\mathbf{d}`
    can be expressed as

    .. math::

        y_i = d_i x_i  \quad \forall i=1,2,...,N

    This is equivalent to a matrix-vector multiplication with a matrix
    containing the vector :math:`\mathbf{d}` along its main diagonal.

    For real-valued ``diag``, the Diagonal operator is self-adjoint as the
    adjoint of a diagonal matrix is the diagonal matrix itself. For
    complex-valued ``diag``, the adjoint is equivalent to the element-wise
    multiplication with the complex conjugate elements of ``diag``.

    """
    def __init__(self, diag, dims=None, dir=0, dtype='float64'):
        self.diag = diag.flatten()
        self.complex = True if np.iscomplexobj(self.diag) else False
        if dims is None:
            self.shape = (len(self.diag), len(self.diag))
            self.dims = None
            self.reshape = False
        else:
            diagdims = [1] * len(dims)
            diagdims[dir] = dims[dir]
            self.diag = self.diag.reshape(diagdims)
            self.shape = (np.prod(dims), np.prod(dims))
            self.dims = dims
            self.reshape = True
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            y = self.diag * x.ravel()
        else:
            x = x.reshape(self.dims)
            y = self.diag * x
        return y.ravel()

    def _rmatvec(self, x):
        if self.complex:
            diagadj = self.diag.conj()
        else:
            diagadj = self.diag
        if not self.reshape:
            y = diagadj * x.ravel()
        else:
            x = x.reshape(self.dims)
            y = diagadj * x
        return y.ravel()

    def matrix(self):
        """Return diagonal matrix as dense :obj:`numpy.ndarray`

        Returns
        ----------
        densemat : :obj:`numpy.ndarray`
            Dense matrix.

        """
        densemat = np.diag(self.diag.squeeze())
        return densemat
