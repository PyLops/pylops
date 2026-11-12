from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import TYPE_CHECKING

import numpy as np

from pylops.utils.typing import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator as spLinearOperator

    from pylops.linearoperator import LinearOperator


def _value_or_sized_to_array(value_or_sized, repeat: int = 1) -> NDArray:
    """Convert an object which is either single value or a list-like to an array.

    Parameters
    ----------
    value_or_sized : `obj`:`int` or `obj`:`float` or `obj`:`list` or `obj`:`tuple`
        Single value or list-like.
    repeat : `obj`:`int`
        Size of resulting array if value is passed. If list is passed, it is ignored.

    Returns
    -------
    out : `obj`:`numpy.array`
        When the input is a single value, returned an array with `repeat` samples
        containing that value. When the input is a list-like object, converts it to an
        array.

    """
    return (
        np.asarray(value_or_sized)
        if isinstance(value_or_sized, Sized)
        else np.array([value_or_sized] * repeat)
    )


def _value_or_sized_to_tuple(value_or_sized, repeat: int = 1) -> tuple:
    """Convert an object which is either single value or a list-like to a tuple.

    Parameters
    ----------
    value_or_sized : `obj`:`int` or `obj`:`float` or `obj`:`list` or `obj`:`tuple`
        Single value or list-like.
    repeat : `obj`:`int`
        Size of resulting array if value is passed. If list is passed, it is ignored.

    Returns
    -------
    out : `obj`:`tuple`
        When the input is a single value, returned an array with `repeat` samples
        containing that value. When the input is a list-like object, converts it to a
        tuple.

    """
    return tuple(_value_or_sized_to_array(value_or_sized, repeat=repeat))


def _raise_on_wrong_dtype(arr: ArrayLike, dtype: DTypeLike, name: str) -> None:
    """Raises an error if dtype of `arr` is not a subdtype of `dtype`.

    Parameters
    ----------
    arr : `obj`:`numpy.array`
        Array whose type will be checked
    dtype : `obj`:`numpy.dtype`
        Type which must be a supertype of `arr.dtype`.
    name : `obj`:`str`
        Name of parameter to issue error.

    Raises
    ------
    TypeError
        When `arr.dtype` is not a subdtype of `dtype`.

    """
    if not np.issubdtype(arr.dtype, dtype):
        msg = f"Wrong input type for `{name}`. Must be {dtype}, but received to {arr.dtype}."
        raise TypeError(msg)


def _get_dtype(
    operators: Sequence[LinearOperator | spLinearOperator | None],
    dtypes: Sequence[DTypeLike] | None = None,
) -> np.dtype:
    """Infer the dtype resulting from combining operators and additional dtypes.

    Parameters
    ----------
    operators : `obj`:`list`
        Operators whose ``dtype`` attribute is used in the inference (objects
        that are ``None`` or have no ``dtype`` attribute are skipped).
    dtypes : `obj`:`list`, optional
        Additional ``numpy.dtype``-like objects used in the inference.

    Returns
    -------
    dtype : `obj`:`numpy.dtype`
        Resulting dtype, obtained with ``np.result_type`` after casting every
        input to a ``numpy.dtype`` (e.g., strings like ``"float64"`` are
        interpreted as dtype names and not as string scalars).

    """
    dtypes = [] if dtypes is None else list(dtypes)
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype"):
            dtypes.append(obj.dtype)
    return np.result_type(*[np.dtype(dtype) for dtype in dtypes])
