__all__ = [
    "NDArray",
    "ShapeLike",
]

from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

IntNDArray = npt.NDArray[np.int_]
NDArray = npt.NDArray

InputDimsLike = Union[Sequence[int], IntNDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike
