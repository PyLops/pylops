import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module


class Imag(LinearOperator):
    r"""Imag operator.

    Return the imaginary component of the input as a real value.
    The adjoint returns a complex number with zero real component and
    the imaginary component set to the real component of the input.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
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

    Notes
    -----
    In forward mode:

    .. math::

        y_{i} = \Im\{x_{i}\} \quad \forall i=0,\ldots,N-1

    In adjoint mode:

    .. math::

        x_{i} = 0 + i\Re\{y_{i}\} \quad \forall i=0,\ldots,N-1

    """

    def __init__(self, dims, dtype="complex128"):
        self.shape = (np.prod(np.array(dims)), np.prod(np.array(dims)))
        self.dtype = np.dtype(dtype)
        self.rdtype = np.real(np.ones(1, self.dtype)).dtype
        self.explicit = False
        self.clinear = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        return ncp.imag(x).astype(self.rdtype)

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        return (0 + 1j * ncp.real(x)).astype(self.dtype)
