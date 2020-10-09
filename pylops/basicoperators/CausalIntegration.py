import numpy as np
from pylops import LinearOperator


class CausalIntegration(LinearOperator):
    r"""Causal integration.

    Apply causal integration to a multi-dimensional array along ``dir`` axis.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    halfcurrent : :obj:`float`, optional
        Add half of current value (``True``) or the entire value (``False``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``)
        or not (``False``)

    Notes
    -----
    The CausalIntegration operator applies a causal integration to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the causal integration is:

    .. math::
        y(t) = \int x(t) dt

    which can be discretised as :

    .. math::
        y[i] = \sum_{j=0}^i x[j] dt

    or

    .. math::
        y[i] = (\sum_{j=0}^{i-1} x[j] + 0.5x[i]) dt

    where :math:`dt` is the ``sampling`` interval. In our implementation, the
    choice to add :math:`x[i]` or just :math:`0.5x[i]` is made by selecting
    the ``halfcurrent`` parameter.

    Note that the integral of a signal has no unique solution, as any constant
    :math:`c` can be added to :math:`y`, for example if :math:`x(t)=t^2` the
    resulting integration is:

    .. math::
        y(t) = \int t^2 dt = \frac{t^3}{3} + c

    If we apply a first derivative to :math:`y` we in fact obtain:

    .. math::
        x(t) = \frac{dy}{dt} = t^2

    no matter the choice of :math:`c`.

    """
    def __init__(self, N, dims=None, dir=-1, sampling=1,
                 halfcurrent=True, dtype='float64'):
        self.N = N
        self.dir = dir
        self.sampling = sampling
        self.halfcurrent = halfcurrent
        if dims is None:
            self.dims = [self.N, 1]
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = dims
                self.reshape = True
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        if self.dir != -1:
            x = np.swapaxes(x, self.dir, -1)
        y = self.sampling * np.cumsum(x, axis=-1)
        if self.halfcurrent:
            y -= self.sampling * x / 2.
        if self.dir != -1:
            y = np.swapaxes(y, -1, self.dir)
        return y.ravel()

    def _rmatvec(self, x):
        if self.reshape:
            x = np.reshape(x, self.dims)
        if self.dir != -1:
            x = np.swapaxes(x, self.dir, -1)
        xflip = np.flip(x, axis=-1)
        if self.halfcurrent:
            y = self.sampling * (np.cumsum(xflip, axis=-1) - xflip/2.)
        else:
            y = self.sampling * np.cumsum(xflip, axis=-1)
        y = np.flip(y, axis=-1)

        if self.dir != -1:
            y = np.swapaxes(y, -1, self.dir)
        return y.ravel()
