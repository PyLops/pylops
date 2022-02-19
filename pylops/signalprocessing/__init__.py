"""
Signal processing
=================

The subpackage signalprocessing provides linear operators for several signal
processing algorithms with forward and adjoint functionalities.

A list of operators present in pylops.signalprocessing:

    Convolve1D	                    1D convolution operator.
    Convolve2D	                    2D convolution operator.
    ConvolveND	                    ND convolution operator.
    Interp          	            Interpolation operator.
    Bilinear                    	Bilinear interpolation operator.
    FFT                             One dimensional Fast-Fourier Transform.
    FFT2D                       	Two dimensional Fast-Fourier Transform.
    FFTND                           N-dimensional Fast-Fourier Transform.
    Shift                           Fractional Shift operator.
    DWT                             One dimensional Wavelet operator.
    DWT2D                           Two dimensional Wavelet operator.
    Seislet                         Two dimensional Seislet operator.
    Radon2D	                        Two dimensional Radon transform.
    Radon3D	                        Three dimensional Radon transform.
    ChirpRadon2D	                Two dimensional Chirp Radon transform.
    ChirpRadon3D	                Three dimensional Chirp Radon transform.
    Sliding1D	                    1D Sliding transform operator.
    Sliding2D	                    2D Sliding transform operator.
    Sliding3D	                    3D Sliding transform operator.
    Patch2D	                        2D Patching transform operator.
    Fredholm1	                    Fredholm integral of first kind.

"""

from .Bilinear import Bilinear
from .ChirpRadon2D import ChirpRadon2D
from .ChirpRadon3D import ChirpRadon3D
from .Convolve1D import Convolve1D
from .Convolve2D import Convolve2D
from .ConvolveND import ConvolveND
from .DWT import DWT
from .DWT2D import DWT2D
from .FFT import FFT
from .FFT2D import FFT2D
from .FFTND import FFTND
from .Fredholm1 import Fredholm1
from .Interp import Interp
from .Patch2D import Patch2D
from .Radon2D import Radon2D
from .Radon3D import Radon3D
from .Seislet import Seislet
from .Shift import Shift
from .Sliding1D import Sliding1D
from .Sliding2D import Sliding2D
from .Sliding3D import Sliding3D

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
