"""
Wave Equation processing
========================

The subpackage waveeqprocessing provides linear operators and applications
aimed at solving various inverse problems in the area of Seismic Wave
Equation Processing.

A list of operators present in pylops.waveeqprocessing:

    PressureToVelocity              Pressure to Vertical velocity conversion.
    UpDownComposition2D             2D Up-down wavefield composition.
    UpDownComposition3D             3D Up-down wavefield composition.
    MDC                             Multi-dimensional convolution.
    PhaseShift                      Phase shift operator.
    Blending                        Blending operator.
    Kirchhoff                       Kirchoff demigration operator.
    AcousticWave2D                  Two-way wave equation demigration operator.

and a list of applications:

    SeismicInterpolation            Seismic interpolation (or regularization).
    Deghosting                      Single-component wavefield decomposition.
    WavefieldDecomposition          Multi-component wavefield decomposition.
    MDD                             Multi-dimensional deconvolution.
    Marchenko                       Marchenko redatuming.
    LSM                             Least-squares Migration (LSM).

"""

from .blending import *
from .kirchhoff import *
from .lsm import *
from .marchenko import *
from .mdd import *
from .oneway import *
from .seismicinterpolation import *
from .twoway import *
from .wavedecomposition import *

__all__ = [
    "MDC",
    "MDD",
    "Marchenko",
    "SeismicInterpolation",
    "PressureToVelocity",
    "UpDownComposition2D",
    "UpDownComposition3D",
    "WavefieldDecomposition",
    "PhaseShift",
    "Deghosting",
    "Blending",
    "Kirchhoff",
    "AcousticWave2D",
    "LSM",
]
