import numpy as np
from pylops import LinearOperator


class FirstDerivative(LinearOperator):
    r"""First derivative.

    Apply first derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    The FirstDerivative operator applies a first derivative to any chosen
    direction of a multi-dimensional array using either a second-order
    centered stencil or first-order forward/backward stencils.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (0.5x[i+1] - 0.5x[i-1]) / dx

    while the first-order forward stencil is:

    .. math::
        y[i] = (x[i+1] - x[i]) / dx

    and the first-order backward stencil is:

    .. math::
        y[i] = (x[i] - x[i-1]) / dx

    """
    def __init__(self, N, dims=None, dir=0, sampling=1.,
                 edge=False, dtype='float64', kind='centered'):
        self.N = N
        self.sampling = sampling
        self.edge = edge
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N')
            else:
                self.dims = dims
                self.reshape = True
        self.dir = dir if dir >= 0 else len(self.dims) + dir
        self.kind = kind
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # choose _matvec and _rmatvec kind
        if self.kind == 'forward':
            self._matvec = self._matvec_forward
            self._rmatvec = self._rmatvec_forward
        elif self.kind == 'centered':
            self._matvec = self._matvec_centered
            self._rmatvec = self._rmatvec_centered
        elif self.kind == 'backward':
            self._matvec = self._matvec_backward
            self._rmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError('kind must be forward, centered, '
                                      'or backward')

    def _matvec_forward(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[:-1] = (x[1:] - x[:-1]) / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[:-1] = (x[1:] - x[:-1]) / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec_forward(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[:-1] -= x[:-1] / self.sampling
            y[1:] += x[:-1] / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[:-1] -= x[:-1] / self.sampling
            y[1:] += x[:-1] / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _matvec_centered(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
            if self.edge:
                y[0] = (x[1] - x[0]) / self.sampling
                y[-1] = (x[-1] - x[-2]) / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
            if self.edge:
                y[0] = (x[1] - x[0]) / self.sampling
                y[-1] = (x[-1] - x[-2]) / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec_centered(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[0:-2] -= (0.5 * x[1:-1]) / self.sampling
            y[2:] += (0.5 * x[1:-1]) / self.sampling
            if self.edge:
                y[0] -= x[0] / self.sampling
                y[1] += x[0] / self.sampling
                y[-2] -= x[-1] / self.sampling
                y[-1] += x[-1] / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[0:-2] -= (0.5 * x[1:-1]) / self.sampling
            y[2:] += (0.5 * x[1:-1]) / self.sampling
            if self.edge:
                y[0] -= x[0] / self.sampling
                y[1] += x[0] / self.sampling
                y[-2] -= x[-1] / self.sampling
                y[-1] += x[-1] / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _matvec_backward(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec_backward(self, x):
        if not self.reshape:
            x = x.squeeze()
            y = np.zeros(self.N, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
        else:
            x = np.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = np.swapaxes(x, self.dir, 0)
            y = np.zeros(x.shape, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
            if self.dir > 0:
                y = np.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
