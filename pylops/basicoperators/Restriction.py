import numpy as np
import numpy.ma as np_ma

from pylops import LinearOperator


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

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
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    Extraction (or *sampling*) of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = x_{l_i}  \quad \forall i=1,2,...,M

    where :math:`\mathbf{L}=[l_1, l_2, l_M]` is a vector containing the indeces
    of the original array at which samples are taken.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`\mathbf{L}=[l_1, l_2, l_M]` in the model vector:

    .. math::

        x_{l_i} = y_i  \quad \forall i=1,2,...,M

    and :math:`x_{j}=0 j \neq l_i` (i.e., at all other locations in input
    vector).

    """
    def __init__(self, M, iava, dtype='float64'):
        self.M = M
        self.iava = iava
        self.shape = (len(iava), M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return x[self.iava]

    def _rmatvec(self, x):
        y = np.zeros(self.M, dtype=self.dtype)
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
        y = np_ma.array(np.zeros(self.M), mask=np.ones(self.M),
                        dtype=self.dtype)
        y.mask[self.iava] = False
        y[self.iava] = x[self.iava]
        return y
