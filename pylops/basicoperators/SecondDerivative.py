import warnings

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils.backend import get_array_module


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply a second derivative using a three-point stencil finite-difference
    approximation along ``axis``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which derivative is applied.
    dir : :obj:`int`, optional

        .. deprecated:: 2.0.0
            Use ``axis`` instead. Note that the default for ``axis`` is -1
            instead of 0 which was the default for ``dir``.

    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta x`.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``)
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
    The SecondDerivative operator applies a second derivative to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / \Delta x^2

    """

    def __init__(
        self, N, dims=None, axis=-1, dir=None, sampling=1, edge=False, dtype="float64"
    ):
        self.N = N
        self.sampling = sampling
        self.edge = edge
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError("product of dims must equal N!")
            else:
                self.dims = dims
                self.reshape = True
        if dir is not None:
            warnings.warn(
                "dir will be deprecated in version 2.0.0, use axis instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            axis = dir
        else:
            axis = axis
        self.axis = normalize_axis_index(axis, len(self.dims))
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[1:-1] = (x[2:] - 2 * x[1:-1] + x[0:-2]) / self.sampling ** 2
            if self.edge:
                y[0] = (x[0] - 2 * x[1] + x[2]) / self.sampling ** 2
                y[-1] = (x[-3] - 2 * x[-2] + x[-1]) / self.sampling ** 2
        else:
            x = ncp.reshape(x, self.dims)
            if self.axis > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.axis, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[1:-1] = (x[2:] - 2 * x[1:-1] + x[0:-2]) / self.sampling ** 2
            if self.edge:
                y[0] = (x[0] - 2 * x[1] + x[2]) / self.sampling ** 2
                y[-1] = (x[-3] - 2 * x[-2] + x[-1]) / self.sampling ** 2
            if self.axis > 0:
                y = ncp.swapaxes(y, 0, self.axis)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[0:-2] += (x[1:-1]) / self.sampling ** 2
            y[1:-1] -= (2 * x[1:-1]) / self.sampling ** 2
            y[2:] += (x[1:-1]) / self.sampling ** 2
            if self.edge:
                y[0] += x[0] / self.sampling ** 2
                y[1] -= 2 * x[0] / self.sampling ** 2
                y[2] += x[0] / self.sampling ** 2
                y[-3] += x[-1] / self.sampling ** 2
                y[-2] -= 2 * x[-1] / self.sampling ** 2
                y[-1] += x[-1] / self.sampling ** 2
        else:
            x = ncp.reshape(x, self.dims)
            if self.axis > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.axis, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[0:-2] += (x[1:-1]) / self.sampling ** 2
            y[1:-1] -= (2 * x[1:-1]) / self.sampling ** 2
            y[2:] += (x[1:-1]) / self.sampling ** 2
            if self.edge:
                y[0] += x[0] / self.sampling ** 2
                y[1] -= 2 * x[0] / self.sampling ** 2
                y[2] += x[0] / self.sampling ** 2
                y[-3] += x[-1] / self.sampling ** 2
                y[-2] -= 2 * x[-1] / self.sampling ** 2
                y[-1] += x[-1] / self.sampling ** 2
            if self.axis > 0:
                y = ncp.swapaxes(y, 0, self.axis)
            y = y.ravel()
        return y
