"""
Signal processing
=================

The subpackage signalprocessing provides linear operators for several signal
processing algorithms with forward and adjoint functionalities.

A list of operators present in pylops.signalprocessing:

    Convolve1D                      1D convolution operator.
    Convolve2D                      2D convolution operator.
    ConvolveND                      ND convolution operator.
    NonStationaryConvolve1D         1D nonstationary convolution operator.
    NonStationaryConvolve2D         2D nonstationary convolution operator.
    NonStationaryConvolve3D         3D nonstationary convolution operator.
    NonStationaryFilters1D          1D nonstationary filter estimation operator.
    NonStationaryFilters2D          2D nonstationary filter estimation operator.
    Interp                          Interpolation operator.
    Bilinear                        Bilinear interpolation operator.
    FFT                             One dimensional Fast-Fourier Transform.
    FFT2D                           Two dimensional Fast-Fourier Transform.
    FFTND                           N-dimensional Fast-Fourier Transform.
    Shift                           Fractional Shift operator.
    DWT                             One dimensional Wavelet operator.
    DWT2D                           Two dimensional Wavelet operator.
    DWTND                           N-dimensional Wavelet operator.
    DCT                             Discrete Cosine Transform.
    DTCWT                           Dual-Tree Complex Wavelet Transform.
    Radon2D	                        Two dimensional Radon transform.
    Radon3D	                        Three dimensional Radon transform.
    Seislet                         Two dimensional Seislet operator.
    Sliding1D                       1D Sliding transform operator.
    Sliding2D                       2D Sliding transform operator.
    Sliding3D                       3D Sliding transform operator.
    Patch2D                         2D Patching transform operator.
    Patch3D                         3D Patching transform operator.
    Fredholm1                       Fredholm integral of first kind.

"""

from .fft import *
from .fft2d import *
from .fftnd import *
from .convolve1d import *
from .convolvend import *
from .convolve2d import *
from .nonstatconvolve1d import *
from .nonstatconvolve2d import *
from .nonstatconvolve3d import *
from .shift import *
from .interp import *
from .bilinear import *
from .radon2d import *
from .radon3d import *
from .chirpradon2d import *
from .chirpradon3d import *
from .sliding1d import *
from .sliding2d import *
from .sliding3d import *
from .patch2d import *
from .patch3d import *
from .fredholm1 import *
from .dwt import *
from .dwt2d import *
from .dwtnd import *
from .seislet import *
from .dct import *
from .dtcwt import *


__all__ = [
    "FFT",
    "FFT2D",
    "FFTND",
    "Shift",
    "Convolve1D",
    "ConvolveND",
    "Convolve2D",
    "NonStationaryConvolve1D",
    "NonStationaryConvolve2D",
    "NonStationaryConvolve3D",
    "NonStationaryFilters1D",
    "NonStationaryFilters2D",
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
    "Patch3D",
    "Fredholm1",
    "DWT",
    "DWT2D",
    "DWTND",
    "Seislet",
    "DCT",
    "DTCWT",
]
