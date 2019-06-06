import numpy as np
from pylops import LinearOperator


class Zero(LinearOperator):
    r"""Zero operator.

    Transform model into array of zeros of size :math:`N` in forward
    and transform data into array of zeros of size :math:`N` in adjoint.

    Parameters
    ----------
    N : :obj:`int`
       Number of samples in data (and model in M is not provided).
    M : :obj:`int`, optional
       Number of samples in model.
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
    An *Zero* operator simply creates a null data vector :math:`\mathbf{y}` in
    forward mode:

    .. math::

       \mathbf{0} \mathbf{x} = \mathbf{0}_N

    and a null model vector :math:`\mathbf{x}` in forward mode:

    .. math::

       \mathbf{0} \mathbf{y} = \mathbf{0}_M

    """
    def __init__(self, N, M=None, dtype='float64'):
        M = N if M is None else M
        self.shape = (N, M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return np.zeros(self.shape[0], dtype=self.dtype)

    def _rmatvec(self, x):
        return np.zeros(self.shape[1], dtype=self.dtype)
