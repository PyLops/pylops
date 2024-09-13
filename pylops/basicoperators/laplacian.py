__all__ = ["Laplacian"]


from typing import Tuple

from pylops import LinearOperator
from pylops.basicoperators import SecondDerivative
from pylops.utils.backend import get_normalize_axis_index
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Laplacian(LinearOperator):
    r"""Laplacian.

    Apply second-order centered Laplacian operator to a multi-dimensional array.

    .. note:: At least 2 dimensions are required, use
      :py:func:`pylops.SecondDerivative` for 1d arrays.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which the Laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=(1, 1)``)
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``) for centered derivative
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Raises
    ------
    ValueError
        If ``axes``. ``weights``, and ``sampling`` do not have the same size

    Notes
    -----
    The Laplacian operator applies a second derivative along two directions of
    a multi-dimensional array.

    For simplicity, given a two dimensional array, the Laplacian is:

    .. math::
        y[i, j] = (x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j])
                  / (\Delta x \Delta y)

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (-2, -1),
        weights: Tuple[float, ...] = (1, 1),
        sampling: Tuple[float, ...] = (1, 1),
        edge: bool = False,
        kind: str = "centered",
        dtype: DTypeLike = "float64",
        name: str = "L",
    ):
        axes = tuple(get_normalize_axis_index()(ax, len(dims)) for ax in axes)
        if not (len(axes) == len(weights) == len(sampling)):
            raise ValueError("axes, weights, and sampling have different size")
        self.axes = axes
        self.weights = weights
        self.sampling = sampling
        self.edge = edge
        self.kind = kind
        Op = self._calc_l2op(
            dims=dims,
            axes=axes,
            sampling=sampling,
            edge=edge,
            kind=kind,
            dtype=dtype,
            weights=weights,
        )
        super().__init__(Op=Op, name=name)

    def _matvec(self, x: NDArray) -> NDArray:
        return super()._matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return super()._rmatvec(x)

    @staticmethod
    def _calc_l2op(
        dims: InputDimsLike,
        axes: InputDimsLike,
        weights: Tuple[float, ...],
        sampling: Tuple[float, ...],
        edge: bool,
        kind: str,
        dtype: DTypeLike,
    ):
        l2op = SecondDerivative(
            dims, axis=axes[0], sampling=sampling[0], edge=edge, kind=kind, dtype=dtype
        )
        dims = l2op.dims
        l2op *= weights[0]
        for ax, samp, weight in zip(axes[1:], sampling[1:], weights[1:]):
            l2op += weight * SecondDerivative(
                dims, axis=ax, sampling=samp, edge=edge, dtype=dtype
            )
        return l2op
