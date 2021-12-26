from . import (
    avo,
    basicoperators,
    optimization,
    signalprocessing,
    utils,
    waveeqprocessing,
)
from .avo.poststack import PoststackLinearModelling
from .avo.prestack import PrestackLinearModelling, PrestackWaveletModelling
from .basicoperators import (
    Block,
    BlockDiag,
    CausalIntegration,
    Conj,
    Diagonal,
    FirstDerivative,
    FirstDirectionalDerivative,
    Flip,
    FunctionOperator,
    Gradient,
    HStack,
    Identity,
    Imag,
    Kronecker,
    Laplacian,
    LinearRegression,
    MatrixMult,
    MemoizeOperator,
    Pad,
    Real,
    Regression,
    Restriction,
    Roll,
    SecondDerivative,
    SecondDirectionalDerivative,
    Smoothing1D,
    Smoothing2D,
    Spread,
    Sum,
    Symmetrize,
    Transpose,
    VStack,
    Zero,
)
from .LinearOperator import LinearOperator
from .optimization.leastsquares import (
    NormalEquationsInversion,
    PreconditionedInversion,
    RegularizedInversion,
)
from .optimization.solver import cg, cgls
from .optimization.sparsity import FISTA, IRLS, ISTA, OMP, SPGL1, SplitBregman
from .utils.deps import *
from .utils.seismicevents import linear2d, makeaxis, parabolic2d
from .utils.tapers import cosinetaper, hanningtaper, taper2d, taper3d
from .utils.utils import Report
from .utils.wavelets import gaussian, ricker

try:
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. pylops should be installed
    # properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")
