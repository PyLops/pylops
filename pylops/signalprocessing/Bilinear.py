import logging
import numpy as np
from pylops import LinearOperator

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

def _checkunique(iava):
    _, count = np.unique(iava, axis=1, return_counts=True)
    if np.any(count > 1):
        raise ValueError('Repeated values in iava array')


class Bilinear(LinearOperator):
    r"""Bilinear interpolation operator.

    Apply bilinear interpolation onto fractionary positions ``iava``
    along the first two axes of a n-dimensional array.

    .. note:: The vector ``iava`` should contain unique pais. If the same
       pair is repeated twice an error will be raised.

    Parameters
    ----------
    iava : :obj:`list` or :obj:`numpy.ndarray`
         Array of size :math:`[2 \times n_{ava}]` containing
         pairs of floating indices of locations of available samples
         for interpolation.
    dims : :obj:`list`
        Number of samples for each dimension
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    ValueError
        If the vector ``iava`` contains repeated values.

    Notes
    -----
    Bilinear interpolation of a subset of :math:`N` values at locations
    ``iava`` from an input n-dimensional vector :math:`\mathbf{x}` of size
    :math:`[m_1 \times m_2 \times ... \times m_{ndim}]` can be expressed as:

    .. math::

        y_{\mathbf{i}} = (1-w^0_{i}) (1-w^1_{i}) x_{l^{l,0}_i, l^{l,1}_i} +
            w^0_{i} (1-w^1_{i}) x_{l^{r,0}_i, l^{l,1}_i} +
            (1-w^0_{i}) w^1_{i} x_{l^{l,0}_i, l^{r,1}_i} +
            w^0_{i} w^1_{i} x_{l^{r,0}_i, l^{r,1}_i}
        \quad \forall i=1,2,...,M

    where :math:`\mathbf{l^{l,0}}=[\lfloor l_1^0 \rfloor,
    \lfloor l_2^0 \rfloor, ..., \lfloor l_N^0 \rfloor]`,
    :math:`\mathbf{l^{l,1}}=[\lfloor l_1^1 \rfloor,
    \lfloor l_2^1 \rfloor, ..., \lfloor l_N^1 \rfloor]`,
    :math:`\mathbf{l^{r,0}}=[\lfloor l_1^0 \rfloor + 1,
    \lfloor l_2^0 \rfloor + 1, ..., \lfloor l_N^0 \rfloor + 1]`,
    :math:`\mathbf{l^{r,1}}=[\lfloor l_1^1 \rfloor + 1,
    \lfloor l_2^1 \rfloor + 1, ..., \lfloor l_N^1 \rfloor + 1]`,
    are vectors containing the indices of the original array at which samples
    are taken, and :math:`\mathbf{w^j}=[l_1^i - \lfloor l_1^i \rfloor,
    l_2^i - \lfloor l_2^i \rfloor, ..., l_N^i - \lfloor l_N^i \rfloor]`
    (:math:`\forall j=0,1`) are the bilinear interpolation weights.

    """
    def __init__(self, iava, dims, dtype='float64'):
        # check non-unique pairs
        _checkunique(iava)

        # define dimension of data
        ndims = len(dims)
        self.dims = dims
        self.dimsd = [len(iava[1])] + list(dims[2:])

        # find indices and weights
        self.iava_t = np.floor(iava[0]).astype(np.int)
        self.iava_b = self.iava_t + 1
        self.weights_tb = iava[0] - self.iava_t
        self.iava_l = np.floor(iava[1]).astype(np.int)
        self.iava_r = self.iava_l + 1
        self.weights_lr = iava[1] - self.iava_l

        # expand dims to weights for nd-arrays
        if ndims > 2:
            for _ in range(ndims - 2):
                self.weights_tb = \
                    np.expand_dims(self.weights_tb, axis=-1)
                self.weights_lr = \
                    np.expand_dims(self.weights_lr, axis=-1)

        self.shape = (np.prod(np.array(self.dimsd)),
                      np.prod(np.array(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = x[self.iava_t, self.iava_l] * (1 - self.weights_tb) * (1 - self.weights_lr) + \
            x[self.iava_t, self.iava_r] * (1 - self.weights_tb) * self.weights_lr + \
            x[self.iava_b, self.iava_l] * self.weights_tb * (1 - self.weights_lr) + \
            x[self.iava_b, self.iava_r] * self.weights_tb * self.weights_lr

        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dimsd)
        y = np.zeros(self.dims, dtype=self.dtype)
        np.add.at(y, [self.iava_t, self.iava_l],
                  x * (1 - self.weights_tb) * (1 - self.weights_lr))
        np.add.at(y, [self.iava_t, self.iava_r],
                  x * (1 - self.weights_tb) * self.weights_lr)
        np.add.at(y, [self.iava_b, self.iava_l],
                  x * self.weights_tb * (1 - self.weights_lr))
        np.add.at(y, [self.iava_b, self.iava_r],
                  x * self.weights_tb * self.weights_lr)
        y = y.ravel()
        return y
