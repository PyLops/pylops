import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.backend import get_array_module


class Real(LinearOperator):
    r"""Real operator.

    Return the real component of the input. The adjoint returns a complex
    number with the same real component as the input and zero imaginary
    component.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    In forward mode:

    .. math::

        y_{i} = \Re\{x_{i}\} \quad \forall i=0,\ldots,N-1

    In adjoint mode:

    .. math::

        x_{i} = \Re\{y_{i}\} + 0i \quad \forall i=0,\ldots,N-1

    """

    def __init__(self, dims, dtype="complex128", name="R"):
        self.dims = self.dimsd = _value_or_list_like_to_tuple(dims)

        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.rdtype = np.real(np.ones(1, self.dtype)).dtype
        super().__init__(explicit=False, clinear=False, name=name)

    def _matvec(self, x):
        ncp = get_array_module(x)
        return ncp.real(x).astype(self.rdtype)

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        return (ncp.real(x) + 0j).astype(self.dtype)
