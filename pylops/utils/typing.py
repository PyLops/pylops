__all__ = [
    "IntNDArray",
    "NDArray",
    "InputDimsLike",
    "SamplingLike",
    "ShapeLike",
    "DTypeLike",
    "TensorTypeLike",
]

from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops.utils.deps import torch_enabled

if torch_enabled:
    import torch

IntNDArray = npt.NDArray[np.int_]
NDArray = npt.NDArray

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], NDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None
