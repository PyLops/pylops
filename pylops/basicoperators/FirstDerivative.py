import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.backend import get_array_module


class FirstDerivative(LinearOperator):
    r"""First derivative.

    Apply a first derivative using a three-point stencil finite-difference
    approximation.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which derivative is applied.
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta x`.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``) for centered derivative
    dtype : :obj:`str`, optional
        Type of elements in input array.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
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
    The FirstDerivative operator applies a first derivative to any chosen
    direction of a multi-dimensional array using either a second-order
    centered stencil or first-order forward/backward stencils.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (0.5x[i+1] - 0.5x[i-1]) / \Delta x

    while the first-order forward stencil is:

    .. math::
        y[i] = (x[i+1] - x[i]) / \Delta x

    and the first-order backward stencil is:

    .. math::
        y[i] = (x[i] - x[i-1]) / \Delta x

    """

    def __init__(
        self,
        dims,
        axis=-1,
        sampling=1.0,
        edge=False,
        dtype="float64",
        kind="centered",
        name="F",
    ):
        self.dims = self.dimsd = _value_or_list_like_to_tuple(dims)
        self.axis = normalize_axis_index(axis, len(self.dims))
        self.sampling = sampling
        self.edge = edge
        self.kind = kind

        # choose _matvec and _rmatvec kind
        if self.kind == "forward":
            self._matvec = self._matvec_forward
            self._rmatvec = self._rmatvec_forward
        elif self.kind == "centered":
            self._matvec = self._matvec_centered
            self._rmatvec = self._rmatvec_centered
        elif self.kind == "backward":
            self._matvec = self._matvec_backward
            self._rmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError("kind must be forward, centered, " "or backward")

        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        super().__init__(explicit=False, clinear=True, name=name)

    def _matvec_forward(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] = (x[..., 1:] - x[..., :-1]) / self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y

    def _rmatvec_forward(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] -= x[..., :-1]
        y[..., 1:] += x[..., :-1]
        y /= self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y

    def _matvec_centered(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
        if self.edge:
            y[..., 0] = x[..., 1] - x[..., 0]
            y[..., -1] = x[..., -1] - x[..., -2]
        y /= self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y

    def _rmatvec_centered(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] -= 0.5 * x[..., 1:-1]
        y[..., 2:] += 0.5 * x[..., 1:-1]
        if self.edge:
            y[..., 0] -= x[..., 0]
            y[..., 1] += x[..., 0]
            y[..., -2] -= x[..., -1]
            y[..., -1] += x[..., -1]
        y /= self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y

    def _matvec_backward(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 1:] = (x[..., 1:] - x[..., :-1]) / self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y

    def _rmatvec_backward(self, x):
        ncp = get_array_module(x)
        x = ncp.reshape(x, self.dims)
        x = ncp.swapaxes(x, self.axis, -1)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] -= x[..., 1:]
        y[..., 1:] += x[..., 1:]
        y /= self.sampling
        y = ncp.swapaxes(y, -1, self.axis)
        y = y.ravel()
        return y
