import warnings

import numpy as np

from pylops import LinearOperator


class Flip(LinearOperator):
    r"""Flip along an axis.

    Flip a multi-dimensional array along ``axis``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axis : :obj:`int`, optional
        .. versionadded:: 2.0
        Axis along which model is flipped.
    dir : :obj:`int`, optional
        .. deprecated:: 2.0
            Use ``axis`` instead. Note that the default for ``axis`` is -1
            instead of 0 which was the default for ``dir``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    The Flip operator flips the input model (and data) along any chosen
    direction. For simplicity, given a one dimensional array,
    in forward mode this is equivalent to:

    .. math::
        y[i] = x[N-1-i] \quad \forall i=0,1,2,\ldots,N-1

    where :math:`N` is the lenght of the input model. As this operator is
    self-adjoint, :math:`x` and :math:`y` in the equation above are simply
    swapped in adjoint mode.

    """

    def __init__(self, N, dims=None, axis=-1, dir=None, dtype="float64"):
        self.N = N
        if dir is not None:
            warnings.warn(
                "dir is deprecated in version 2.0, use axis instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            self.axis = dir
        else:
            self.axis = axis
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError("product of dims must equal N")
            else:
                self.dims = dims
                self.reshape = True
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        y = np.flip(x, axis=self.axis)
        if self.reshape:
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        return self._matvec(x)
