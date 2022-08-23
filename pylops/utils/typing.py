__all__ = [
    "NDArrayLike",
    "ShapeLike",
]

from typing import NewType, Tuple

import numpy as np

NDArrayLike = NewType("NDArrayLike", np.ndarray)
ShapeLike = Tuple[int, ...]
