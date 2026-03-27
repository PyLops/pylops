__all__ = ["Interp"]

from typing import Literal, Tuple, Union

import numpy as np

from pylops import LinearOperator, aslinearoperator
from pylops.basicoperators import Diagonal, MatrixMult, Restriction, Transpose
from pylops.signalprocessing._interp_utils import (
    _clip_iava_above_last_sample_index,
    _ensure_iava_is_unique,
)
from pylops.signalprocessing.interpspline import InterpCubicSpline
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.typing import DTypeLike, InputDimsLike, IntNDArray, SamplingLike


def _nearestinterp(
    dims: Union[int, InputDimsLike],
    iava: SamplingLike,
    axis: int = -1,
    dtype: DTypeLike = "float64",
):
    """Nearest neighbour interpolation."""
    iava = np.round(iava).astype(int)
    _ensure_iava_is_unique(iava=iava)
    return Restriction(dims, iava, axis=axis, dtype=dtype), iava


def _linearinterp(
    dims: InputDimsLike,
    iava: SamplingLike,
    axis: int = -1,
    dtype: DTypeLike = "float64",
):
    """Linear interpolation."""
    ncp = get_array_module(iava)

    if np.issubdtype(iava.dtype, np.integer):
        iava = iava.astype(np.float64)

    sample_size = dims[axis]
    dimsd = list(dims)
    dimsd[axis] = len(iava)
    dimsd = tuple(dimsd)

    # ensure that samples are not beyond the last sample, in that case set to
    # penultimate sample and raise a warning
    _clip_iava_above_last_sample_index(iava=iava, sample_size=sample_size)

    # find indices and weights
    iva_l = ncp.floor(iava).astype(int)
    iva_r = iva_l + 1
    weights = iava - iva_l

    # create operators
    Op = Diagonal(1 - weights, dims=dimsd, axis=axis, dtype=dtype) * Restriction(
        dims, iva_l, axis=axis, dtype=dtype
    ) + Diagonal(weights, dims=dimsd, axis=axis, dtype=dtype) * Restriction(
        dims, iva_r, axis=axis, dtype=dtype
    )
    return Op, iava, dims, dimsd


def _sincinterp(
    dims: InputDimsLike,
    iava: SamplingLike,
    axis: int = 0,
    dtype: DTypeLike = "float64",
):
    """Sinc interpolation."""
    ncp = get_array_module(iava)

    _ensure_iava_is_unique(iava=iava)

    # create sinc interpolation matrix
    nreg = dims[axis]
    ireg = ncp.arange(nreg)
    sinc = ncp.tile(iava[:, np.newaxis], (1, nreg)) - ncp.tile(ireg, (len(iava), 1))
    sinc = ncp.sinc(sinc)

    # identify additional dimensions and create MatrixMult operator
    otherdims = np.array(dims)
    otherdims = np.roll(otherdims, -axis)
    otherdims = otherdims[1:]
    Op = MatrixMult(sinc, otherdims=otherdims, dtype=dtype)

    # create Transpose operator that brings axis to first dimension
    dimsd = list(dims)
    dimsd[axis] = len(iava)
    if axis > 0:
        axes = np.arange(len(dims), dtype=int)
        axes = np.roll(axes, -axis)
        Top = Transpose(dims, axes=axes, dtype=dtype)
        T1op = Transpose(dimsd, axes=axes, dtype=dtype)
        Op = T1op.H * Op * Top
    return Op, dims, dimsd


def Interp(
    dims: Union[int, InputDimsLike],
    iava: SamplingLike,
    axis: int = -1,
    kind: Literal["linear", "nearest", "sinc", "cubic_spline"] = "linear",
    dtype: DTypeLike = "float64",
    name: str = "I",
) -> Tuple[LinearOperator, IntNDArray]:
    r"""Interpolation operator.

    Apply interpolation along ``axis`` from regularly sampled input
    vector into fractionary positions ``iava`` using one of the
    following algorithms:

    - *Nearest neighbour* interpolation
      is a thin wrapper around :obj:`pylops.Restriction` at ``np.round(iava)``
      locations.

    - *Linear interpolation* extracts values from input vector
      at locations ``np.floor(iava)`` and ``np.floor(iava)+1`` and linearly
      combines them in forward mode, places weighted versions of the
      interpolated values at locations ``np.floor(iava)`` and
      ``np.floor(iava)+1`` in an otherwise zero vector in adjoint mode.

    - *Sinc interpolation* performs sinc interpolation at locations ``iava``.
      Note that this is the most accurate method but it has higher computational
      cost as it involves multiplying the input data by a matrix of size
      :math:`N \times M`.

    - *Cubic Spline interpolation* relies on a cubic spline, i.e., a 2-times
      continuously differentiable piecewise third order polynomial with equally spaced
      knots. It is interpolated at the locations ``iava`` by evaluating the respective
      polynomial fitted between ``np.floor(iava)`` and ``np.floor(iava) + 1``.
      It offers an excellent tradeoff between accuracy and computational complexity
      and its results oscillate less than those obtained from sinc interpolation.
      It can also be accessed directly via :class:`pylops.singalprocessing.InterpCubicSpline`.

    .. note:: The vector ``iava`` should contain unique values. If the same
      index is repeated twice an error will be raised. This also applies when
      values beyond the last element of the input array for
      *linear interpolation* and *cubic spline interpolation* as those values are forced
      to be just before this element.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    iava : :obj:`list` or :obj:`numpy.ndarray`
         Floating indices of locations of available samples for interpolation.
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which interpolation is applied.
    kind : :obj:`str`, optional
        Kind of interpolation.
        Currently, ``"nearest"``, ``"linear"``, ``"sinc"``, and ``"cubic_spline"`` are
        available.

        .. versionadded:: 2.7.0

        The ``"cubic_spline"``-interpolation was added.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    op : :obj:`pylops.LinearOperator`
        Linear interpolation operator
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Corrected indices of locations of available samples
        (samples at ``M-1`` or beyond are forced to be at ``M-1-eps``)

    Raises
    ------
    ValueError
        If the vector ``iava`` contains repeated values.
    NotImplementedError
        If ``kind`` is not ``nearest``, ``linear`` or ``sinc``

    See Also
    --------
    pylops.Restriction : Restriction operator

    Notes
    -----
    *Linear interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = (1-w_i) x_{l^{l}_i} + w_i x_{l^{r}_i}
        \quad \forall i=1,2,\ldots,N

    where :math:`\mathbf{l^l}=[\lfloor l_1 \rfloor, \lfloor l_2 \rfloor,\ldots,
    \lfloor l_N \rfloor]` and :math:`\mathbf{l^r}=[\lfloor l_1 \rfloor +1,
    \lfloor l_2 \rfloor +1,\ldots,
    \lfloor l_N \rfloor +1]` are vectors containing the indeces
    of the original array at which samples are taken, and
    :math:`\mathbf{w}=[l_1 - \lfloor l_1 \rfloor, l_2 - \lfloor l_2 \rfloor,
    ..., l_N - \lfloor l_N \rfloor]` are the linear interpolation weights.
    This operator can be implemented by simply summing two
    :class:`pylops.Restriction` operators which are weighted
    using :class:`pylops.basicoperators.Diagonal` operators.

    *Sinc interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::
        \DeclareMathOperator{\sinc}{sinc}
        y_i = \sum_{j=0}^{M} x_j \sinc(i-j) \quad \forall i=1,2,\ldots,N

    This operator can be implemented using the :class:`pylops.MatrixMult`
    operator with a matrix containing the values of the sinc function at all
    :math:`i,j` possible combinations.

    """

    kind = kind.lower()  # type: ignore
    if kind not in {"nearest", "linear", "sinc", "cubic_spline"}:
        raise NotImplementedError(f"{kind} interpolation could not be found.")

    dims = _value_or_sized_to_tuple(dims)

    if kind == "nearest":
        interpop, iava = _nearestinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "linear":
        interpop, iava, dims, dimsd = _linearinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "sinc":
        interpop, dims, dimsd = _sincinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "cubic_spline":
        interpop = InterpCubicSpline(
            dims=dims,
            iava=iava,
            axis=axis,
            dtype=dtype,
            name=name,
        )
        iava = interpop.iava

    # add dims and dimsd to composite operators (not needed for neareast as
    # interpop is a Restriction operator already
    if kind not in {"nearest", "cubic_spline"}:
        interpop = aslinearoperator(interpop)
        interpop.dims = dims
        interpop.dimsd = dimsd
        interpop.name = name
    return interpop, iava
