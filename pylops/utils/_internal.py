from typing import Sized, Tuple

import numpy as np
import numpy.typing as npt

from pylops.utils.typing import NDArray, DTypeLike


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


def _value_or_sized_to_tuple(value_or_sized, repeat: int = 1) -> Tuple:
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


def _raise_on_wrong_dtype(arr: npt.ArrayLike, dtype: DTypeLike, name: str) -> None:
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
        raise TypeError(
            (
                f"Wrong input type for `{name}`. "
                f'Must be "{dtype}", but received to "{arr.dtype}".'
            )
        )
