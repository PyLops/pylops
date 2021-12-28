# isort: skip_file
from .lsm import LSM, Demigration
from .marchenko import Marchenko
from .mdd import MDC, MDD
from .oneway import Deghosting, PhaseShift
from .seismicinterpolation import SeismicInterpolation
from .wavedecomposition import (
    PressureToVelocity,
    UpDownComposition2D,
    UpDownComposition3D,
    WavefieldDecomposition,
)
