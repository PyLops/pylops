__all__ = ["SecondDerivative"]

from typing import Callable, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_normalize_axis_index,
    inplace_add,
    inplace_set,
)
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


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
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        sampling: float = 1.0,
        kind: str = "centered",
        edge: bool = False,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axis = get_normalize_axis_index()(axis, len(self.dims))
        self.sampling = sampling
        self.kind = kind
        self.edge = edge
        self.slice = {
            i: {
                j: tuple([slice(None, None)] * (len(dims) - 1) + [slice(i, j)])
                for j in (None, -1, -2, -3, -4)
            }
            for i in (None, 1, 2, 3, 4)
        }
        self.sample = {
            i: tuple([slice(None, None)] * (len(dims) - 1) + [i]) for i in range(-3, 4)
        }
        self._register_multiplications(self.kind)

    def _register_multiplications(
        self,
        kind: str,
    ) -> None:
        # choose _matvec and _rmatvec kind
        self._hmatvec: Callable
        self._hrmatvec: Callable
        if kind == "forward":
            self._hmatvec = self._matvec_forward
            self._hrmatvec = self._rmatvec_forward
        elif kind == "centered":
            self._hmatvec = self._matvec_centered
            self._hrmatvec = self._rmatvec_centered
        elif kind == "backward":
            self._hmatvec = self._matvec_backward
            self._hrmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError(
                "'kind' must be 'forward', 'centered' or 'backward'"
            )

    def _matvec(self, x: NDArray) -> NDArray:
        return self._hmatvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._hrmatvec(x)

    @reshaped(swapaxis=True)
    def _matvec_forward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., :-2] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        y = inplace_set(
            x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2], y, self.slice[None][-2]
        )
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_forward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., :-2] += x[..., :-2]
        y = inplace_add(x[..., :-2], y, self.slice[None][-2])
        # y[..., 1:-1] -= 2 * x[..., :-2]
        y = inplace_add(-2 * x[..., :-2], y, self.slice[1][-1])
        # y[..., 2:] += x[..., :-2]
        y = inplace_add(x[..., :-2], y, self.slice[2][None])
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _matvec_centered(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., 1:-1] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        y = inplace_set(
            x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2], y, self.slice[1][-1]
        )
        if self.edge:
            # y[..., 0] = x[..., 0] - 2 * x[..., 1] + x[..., 2]
            y = inplace_set(x[..., 0] - 2 * x[..., 1] + x[..., 2], y, self.sample[0])
            # y[..., -1] = x[..., -3] - 2 * x[..., -2] + x[..., -1]
            y = inplace_set(
                x[..., -3] - 2 * x[..., -2] + x[..., -1], y, self.sample[-1]
            )
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_centered(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., :-2] += x[..., 1:-1]
        y = inplace_add(x[..., 1:-1], y, self.slice[None][-2])
        # y[..., 1:-1] -= 2 * x[..., 1:-1]
        y = inplace_add(-2 * x[..., 1:-1], y, self.slice[1][-1])
        # y[..., 2:] += x[..., 1:-1]
        y = inplace_add(x[..., 1:-1], y, self.slice[2][None])
        if self.edge:
            # y[..., 0] += x[..., 0]
            y = inplace_add(x[..., 0], y, self.sample[0])
            # y[..., 1] -= 2 * x[..., 0]
            y = inplace_add(-2 * x[..., 0], y, self.sample[1])
            # y[..., 2] += x[..., 0]
            y = inplace_add(x[..., 0], y, self.sample[2])
            # y[..., -3] += x[..., -1]
            y = inplace_add(x[..., -1], y, self.sample[-3])
            # y[..., -2] -= 2 * x[..., -1]
            y = inplace_add(-2 * x[..., -1], y, self.sample[-2])
            # y[..., -1] += x[..., -1]
            y = inplace_add(x[..., -1], y, self.sample[-1])
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _matvec_backward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., 2:] = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        y = inplace_set(
            x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2], y, self.slice[2][None]
        )
        y /= self.sampling**2
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_backward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        # y[..., :-2] += x[..., 2:]
        y = inplace_add(x[..., 2:], y, self.slice[None][-2])
        # y[..., 1:-1] -= 2 * x[..., 2:]
        y = inplace_add(-2 * x[..., 2:], y, self.slice[1][-1])
        # y[..., 2:] += x[..., 2:]
        y = inplace_add(x[..., 2:], y, self.slice[2][None])
        y /= self.sampling**2
        return y
