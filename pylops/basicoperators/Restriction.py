import numpy as np
import numpy.ma as np_ma

from pylops import LinearOperator

class Restriction(LinearOperator):
    """Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava`` in forward mode and
    places those values at locations ``iava`` in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Indeces of available samples for data selection.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    """
    def __init__(self, N, iava, dtype='float32'):
        self.N = N
        self.iava = iava
        self.shape = (len(iava), N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return x[self.iava]

    def _rmatvec(self, x):
        y = np.zeros(self.N, dtype=self.dtype)
        y[self.iava] = x
        return y

    def mask(self, x):
        """Apply mask to input signal returning a signal of same size with
        values at ``iava`` locations and ``0`` at other locations

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array.

        Returns
        ----------
        y : :obj:`numpy.ma.core.MaskedArray`
            Masked array.

        """
        y = np_ma.array(np.zeros(self.N), mask=np.ones(self.N), dtype=self.dtype)
        y.mask[self.iava] = False
        y[self.iava] = x[self.iava]
        return y
