import numpy as np
from pylops import LinearOperator


class Identity(LinearOperator):
    r"""Identity operator.

    Simply move model to data in forward model and viceversa in adjoint mode if :math:`M = N`.
    If :math:`M > N` removes last :math:`M - N` elements from model in forward and pads with
    :math:`0` in adjoint. If :math:`N > M` removes last :math:`N - M` elements from data in adjoint
    and pads with :math:`0` in forward.

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
        Operator contains a matrix that can be solved explicitly (``True``) or not (``False``)

    Notes
    -----
    For :math:`M = N`, an *Identity* operator simply moves the model :math:`\mathbf{x}` to the
    data :math:`\mathbf{y}` in forward mode and viceversa in adjoint mode:

    .. math::

        y_i = x_i  \quad \forall i=1,2,...,N

    or in matrix form:

    .. math::

        \mathbf{y} = \mathbf{I} \mathbf{x} = \mathbf{x}

    and

    .. math::

        \mathbf{x} = \mathbf{I} \mathbf{y} = \mathbf{y}

    For :math:`M > N`, the *Identity* operator takes the first :math:`M` elements of the
    model :math:`\mathbf{x}` into the data :math:`\mathbf{y}` in forward mode

    .. math::

        y_i = x_i  \quad \forall i=1,2,...,N

    and all the elements of the data :math:`\mathbf{y}` into the first :math:`M` elements of
    model in adjoint mode (other elements are ``O``):

    .. math::

        x_i = y_i  \quad \forall i=1,2,...,M

        x_i = 0 \quad \forall i=M+1,...,N

    """
    def __init__(self, N, M=None, dtype=None):
        M = N if M is None else M
        self.shape = (N, M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = x[:self.shape[0]]
        else:
            y = np.zeros(self.shape[0], dtype=self.dtype)
            y[:self.shape[1]] = x
        return y

    def _rmatvec(self, x):
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = np.zeros(self.shape[1], dtype=self.dtype)
            y[:self.shape[0]] = x
        else:
            y = x[:self.shape[1]]
        return y
