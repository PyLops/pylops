"""
Signal processing
=================

The subpackage signalprocessing provides linear operators for several signal
processing algorithms with forward and adjoint functionalities.

A list of operators present in pylops.signalprocessing:

    Convolve1D                      1D convolution operator.
    Convolve2D                      2D convolution operator.
    ConvolveND                      ND convolution operator.
    Interp                          Interpolation operator.
    Bilinear                        Bilinear interpolation operator.
    FFT                             One dimensional Fast-Fourier Transform.
    FFT2D                           Two dimensional Fast-Fourier Transform.
    FFTND                           N-dimensional Fast-Fourier Transform.
    Shift                           Fractional Shift operator.
    DWT                             One dimensional Wavelet operator.
    DWT2D                           Two dimensional Wavelet operator.
    Seislet                         Two dimensional Seislet operator.
    Radon2D	                        Two dimensional Radon transform.
    Radon3D	                        Three dimensional Radon transform.
    Sliding2D                       2D Sliding transform operator.
    Sliding3D                       3D Sliding transform operator.
    Fredholm1                       Fredholm integral of first kind.

"""

from .FFT import *
from .FFT2D import *
from .FFTND import *
from .Convolve1D import *
from .ConvolveND import *
from .Convolve2D import *
from .Shift import *
from .Interp import *
from .Bilinear import *
from .Radon2D import *
from .Radon3D import *
from .ChirpRadon2D import *
from .ChirpRadon3D import *
from .Sliding1D import *
from .Sliding2D import *
from .Sliding3D import *
from .Patch2D import *
from .Fredholm1 import *
from .DWT import *
from .DWT2D import *
from .Seislet import *

__all__ = [
    "FFT",
    "FFT2D",
    "FFTND",
    "Shift",
    "Convolve1D",
    "ConvolveND",
    "Convolve2D",
    "Interp",
    "Bilinear",
    "Radon2D",
    "Radon3D",
    "ChirpRadon2D",
    "ChirpRadon3D",
    "Sliding1D",
    "Sliding2D",
    "Sliding3D",
    "Patch2D",
    "Fredholm1",
    "DWT",
    "DWT2D",
    "Seislet",
]
