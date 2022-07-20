from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt


def _value_or_list_like_to_array(
    value_or_list_like: Union[float, List, Tuple, npt.ArrayLike], repeat: int = 1
) -> npt.ArrayLike:
    """Convert an object which is either single value or a list-like to an array.

    Parameters
    ----------
    value_or_list_like : `obj`:`int` or `obj`:`float` or `obj`:`list` or `obj`:`tuple`
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
    try:
        len(value_or_list_like)
        out = np.array(value_or_list_like)
    except TypeError:
        out = np.array([value_or_list_like] * repeat)
    return out


def _value_or_list_like_to_tuple(
    value_or_list_like: Union[float, List, Tuple, npt.ArrayLike], repeat: int = 1
) -> Tuple:
    """Convert an object which is either single value or a list-like to a tuple.

    Parameters
    ----------
    value_or_list_like : `obj`:`int` or `obj`:`float` or `obj`:`list` or `obj`:`tuple`
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
    return tuple(_value_or_list_like_to_array(value_or_list_like, repeat=repeat))


def _raise_on_wrong_dtype(arr: npt.ArrayLike, dtype: npt.DTypeLike, name: str) -> None:
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
