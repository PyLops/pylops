__all__ = [
    "NDArray",
    "ShapeLike",
]

from typing import Sequence, Tuple

import numpy.typing as npt

InputDimsLike = Sequence[int]
NDArray = npt.NDArray
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike
