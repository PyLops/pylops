from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply a second derivative using a three-point stencil finite-difference
    approximation along ``axis``.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which derivative is applied.
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta x`.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    edge : :obj:`bool`, optional
        Use shifted derivatives at edges (``True``) or
        ignore them (``False``). This is currently only available
         for centered derivative
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
    The SecondDerivative operator applies a second derivative to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / \Delta x^2

    while the second-order forward stencil is:

    .. math::
        y[i] = (x[i+2] - 2x[i+1] + x[i]) / \Delta x^2

    and the second-order backward stencil is:

    .. math::
        y[i] = (x[i] - 2x[i-1] + x[i-2]) / \Delta x^2

    """

    def __init__(
        self,
        dims: Union[int, List],
        axis: int = -1,
        sampling: float = 1.0,
        kind: str = "centered",
        edge: bool = False,
        dtype: str = "float64",
        name: str = "S",
    ) -> None:
        dims = _value_or_list_like_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axis = normalize_axis_index(axis, len(self.dims))
        self.sampling = sampling
        self.kind = kind
        self.edge = edge
        self._register_multiplications(self.kind)

    def _register_multiplications(
        self,
        kind: str,
    ) -> Callable:
        # choose _matvec and _rmatvec kind
        if kind == "forward":
            self._matvec = self._matvec_forward
            self._rmatvec = self._rmatvec_forward
        elif kind == "centered":
            self._matvec = self._matvec_centered
            self._rmatvec = self._rmatvec_centered
        elif kind == "backward":
            self._matvec = self._matvec_backward
            self._rmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError(
                "'kind' must be 'forward', 'centered' or 'backward'"
            )

    @reshaped(swapaxis=True)
    def _matvec_forward(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_forward(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] += x[..., :-2]
        y[..., 1:-1] -= 2 * x[..., :-2]
        y[..., 2:] += x[..., :-2]
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _matvec_centered(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 1:-1] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        if self.edge:
            y[..., 0] = x[..., 0] - 2 * x[..., 1] + x[..., 2]
            y[..., -1] = x[..., -3] - 2 * x[..., -2] + x[..., -1]
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_centered(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] += x[..., 1:-1]
        y[..., 1:-1] -= 2 * x[..., 1:-1]
        y[..., 2:] += x[..., 1:-1]
        if self.edge:
            y[..., 0] += x[..., 0]
            y[..., 1] -= 2 * x[..., 0]
            y[..., 2] += x[..., 0]
            y[..., -3] += x[..., -1]
            y[..., -2] -= 2 * x[..., -1]
            y[..., -1] += x[..., -1]
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _matvec_backward(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 2:] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_backward(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] += x[..., 2:]
        y[..., 1:-1] -= 2 * x[..., 2:]
        y[..., 2:] += x[..., 2:]
        y /= self.sampling**2
        return y
