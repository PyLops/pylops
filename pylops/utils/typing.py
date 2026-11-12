__all__ = [
    "IntNDArray",
    "NDArray",
    "ArrayLike",
    "InputDimsLike",
    "SamplingLike",
    "ShapeLike",
    "DTypeLike",
    "TensorTypeLike",
]

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
from typing import Literal

import numpy as np
import numpy.typing as npt

from pylops.utils.deps import torch_enabled

if torch_enabled:
    import torch

# numpy generic types
NDArray = npt.NDArray[np.number]
ArrayLike = npt.ArrayLike
IntNDArray = npt.NDArray[np.integer]
FloatingNDArray = npt.NDArray[np.floating]
InexactNDArray = npt.NDArray[np.inexact]  # float or complex

InputDimsLike = Sequence[int] | IntNDArray
SamplingLike = Sequence[float] | FloatingNDArray
ShapeLike = tuple[int, ...]
DTypeLike = npt.DTypeLike

# torch generic types
if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None

# pylops specific types
Tctengine = Literal["cpu", "cuda"]
Tengine_nn = Literal["numpy", "numba"]
Tengine_nnc = Literal["numpy", "numba", "cuda"]
Tfftengine_ncj = Literal["numpy", "cupy", "jax"]
Tfftengine_ns = Literal["numpy", "scipy"]
Tfftengine_nsf = Literal["numpy", "scipy", "fftw"]
Tfftengine_nsm = Literal["numpy", "scipy", "mkl_fft"]
Tfftengine_nfsm = Literal["numpy", "fftw", "scipy", "mkl_fft"]
Tinoutengine = tuple[Tfftengine_ncj, Tfftengine_ncj]
Tmriengine = Literal["numpy", "jax"]

Tavolinearization = Literal["akirich", "fatti", "PS"]
Tctprojgeom = Literal["parallel", "fanflat"]
Tctprojectortype = Literal["strip", "line", "linear", "cuda"]
Tderivkind = Literal["forward", "centered", "backward"]
Tfftnorm = Literal["ortho", "none", "1/n"]
Tmrimask = Literal["vertical-reg", "vertical-uni", "radial-reg", "radial-uni"]
Tparallel_kind = Literal["multiproc", "multithread"]
Ttaper = Literal["hanning", "cosine", "cosine_square"]

Tbackend = Literal["numpy", "cupy"]
Tirlskind = Literal["data", "model", "datamodel"]
Tmemunit = Literal["B", "KB", "MB", "GB"]
Tsolverengine = Literal["scipy", "pylops"]
Tthreshkind = Literal[
    "hard",
    "soft",
    "half",
    "hard-percentile",
    "soft-percentile",
    "half-percentile",
]

Tpwdsmoothing = Literal["triangle", "boxcar"]
Tsampler = Literal["gaussian", "rayleigh", "rademacher", "unitvector"]
Tsampler2 = Literal["gaussian", "rayleigh", "rademacher"]

Tpool = Pool | ThreadPoolExecutor
