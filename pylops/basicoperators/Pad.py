import numpy as np
from pylops import LinearOperator


class Pad(LinearOperator):
    r"""Pad operator.

    Zero-pad model in forward model and extract non-zero subsequence
    in adjoint. Padding can be performed in one or multiple directions to any
    multi-dimensional input arrays.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
    pad : :obj:`tuple`
        Number of samples to pad. If ``dims`` is a scalar, ``pad`` is a single
        tuple ``(pad_in, pad_end)``. If ``dims`` is a tuple,
        ``pad`` is a tuple of tuples where each inner tuple contains
        the number of samples to pad in each dimension
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
        If any element of ``pad`` is negative.

    Notes
    -----
    Given an array of size :math:`N`, the *Pad* operator simply adds
    :math:`pad_{in}` at the start and :math:`pad_{end}` at the end in forward mode:

    .. math::

        y_{i} = x_{i-pad_{in}}  \quad \forall
        i=pad_{in},pad_{in}+1,...,pad_{in}+N-1

    and :math:`y_i = 0 \quad \forall
    i=0,...,pad_{in}-1, pad_{in}+N-1,...,N+pad_{in}+pad_{end}`

    In adjoint mode, values from :math:`pad_{in}` to :math:`N-pad_{end}` are
    extracted from the data:

    .. math::

        x_{i} = y_{pad_{in}+i}  \quad \forall i=0, N-1

    """
    def __init__(self, dims, pad, dtype='float64'):
        if np.any(np.array(pad) < 0):
            raise ValueError('Padding must be positive or zero')
        self.dims = dims
        self.pad = pad
        self.reshape = False if isinstance(self.dims, int) else True
        if self.reshape:
            self.dimsd = [dim + p[0] + p[1] for dim, p in zip(dims, pad)]
        else:
            self.dimsd = dims + pad[0] + pad[1]
        self.shape = (np.prod(np.array(self.dimsd)),
                      np.prod(np.array(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            y = x.reshape(self.dims)
            y = np.pad(y, self.pad, mode='constant')
        else:
            y = np.pad(x, self.pad, mode='constant')
        return y.flatten()

    def _rmatvec(self, x):
        if self.reshape:
            y = x.reshape(self.dimsd)
            for ax, pad in enumerate(self.pad):
                y = np.take(y, np.arange(pad[0], pad[0] + self.dims[ax]),
                            axis=ax)
        else:
            y = x[self.pad[0]:self.pad[0]+self.dims]
        return y.flatten()
