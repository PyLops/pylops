__all__ = ["FirstDerivative"]

from typing import Callable, Union

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class FirstDerivative(LinearOperator):
    r"""First derivative.

    Apply a first derivative using a multiple-point stencil finite-difference
    approximation along ``axis``.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which derivative is applied.
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta x`.
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``). This is currently only available
         for centered derivative
    order : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Derivative order (``3`` or ``5``). This is currently only available
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
    The FirstDerivative operator applies a first derivative to any chosen
    direction of a multi-dimensional array using either a second- or third-order
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

    Formulas for the third-order centered stencil can be found at
    this `link <https://en.wikipedia.org/wiki/Finite_difference_coefficient>`_.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        sampling: float = 1.0,
        kind: str = "centered",
        edge: bool = False,
        order: int = 3,
        dtype: DTypeLike = "float64",
        name: str = "F",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axis = normalize_axis_index(axis, len(self.dims))
        self.sampling = sampling
        self.kind = kind
        self.edge = edge
        self.order = order
        self._register_multiplications(self.kind, self.order)

    def _register_multiplications(
        self,
        kind: str,
        order: int,
    ) -> None:
        # choose _matvec and _rmatvec kind
        self._hmatvec: Callable
        self._hrmatvec: Callable
        if kind == "forward":
            self._hmatvec = self._matvec_forward
            self._hrmatvec = self._rmatvec_forward
        elif kind == "centered":
            if order == 3:
                self._hmatvec = self._matvec_centered3
                self._hrmatvec = self._rmatvec_centered3
            elif order == 5:
                self._hmatvec = self._matvec_centered5
                self._hrmatvec = self._rmatvec_centered5
            else:
                raise NotImplementedError("'order' must be '3, or '5'")
        elif kind == "backward":
            self._hmatvec = self._matvec_backward
            self._hrmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError(
                "'kind' must be 'forward', 'centered', or 'backward'"
            )

    def _matvec(self, x: NDArray) -> NDArray:
        return self._hmatvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._hrmatvec(x)

    @reshaped(swapaxis=True)
    def _matvec_forward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] = (x[..., 1:] - x[..., :-1]) / self.sampling
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_forward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] -= x[..., :-1]
        y[..., 1:] += x[..., :-1]
        y /= self.sampling
        return y

    @reshaped(swapaxis=True)
    def _matvec_centered3(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
        if self.edge:
            y[..., 0] = x[..., 1] - x[..., 0]
            y[..., -1] = x[..., -1] - x[..., -2]
        y /= self.sampling
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_centered3(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-2] -= 0.5 * x[..., 1:-1]
        y[..., 2:] += 0.5 * x[..., 1:-1]
        if self.edge:
            y[..., 0] -= x[..., 0]
            y[..., 1] += x[..., 0]
            y[..., -2] -= x[..., -1]
            y[..., -1] += x[..., -1]
        y /= self.sampling
        return y

    @reshaped(swapaxis=True)
    def _matvec_centered5(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 2:-2] = (
            x[..., :-4] / 12.0
            - 2 * x[..., 1:-3] / 3.0
            + 2 * x[..., 3:-1] / 3.0
            - x[..., 4:] / 12.0
        )
        if self.edge:
            y[..., 0] = x[..., 1] - x[..., 0]
            y[..., 1] = 0.5 * (x[..., 2] - x[..., 0])
            y[..., -2] = 0.5 * (x[..., -1] - x[..., -3])
            y[..., -1] = x[..., -1] - x[..., -2]
        y /= self.sampling
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_centered5(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-4] += x[..., 2:-2] / 12.0
        y[..., 1:-3] -= 2.0 * x[..., 2:-2] / 3.0
        y[..., 3:-1] += 2.0 * x[..., 2:-2] / 3.0
        y[..., 4:] -= x[..., 2:-2] / 12.0
        if self.edge:
            y[..., 0] -= x[..., 0] + 0.5 * x[..., 1]
            y[..., 1] += x[..., 0]
            y[..., 2] += 0.5 * x[..., 1]
            y[..., -3] -= 0.5 * x[..., -2]
            y[..., -2] -= x[..., -1]
            y[..., -1] += 0.5 * x[..., -2] + x[..., -1]
        y /= self.sampling
        return y

    @reshaped(swapaxis=True)
    def _matvec_backward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., 1:] = (x[..., 1:] - x[..., :-1]) / self.sampling
        return y

    @reshaped(swapaxis=True)
    def _rmatvec_backward(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(x.shape, self.dtype)
        y[..., :-1] -= x[..., 1:]
        y[..., 1:] += x[..., 1:]
        y /= self.sampling
        return y
