from .LinearOperator import LinearOperator
from .basicoperators import LinearRegression
from .basicoperators import MatrixMult
from .basicoperators import Identity
from .basicoperators import Zero
from .basicoperators import Diagonal

from .basicoperators import VStack
from .basicoperators import HStack
from .basicoperators import BlockDiag

from .basicoperators import CausalIntegration
from .basicoperators import FirstDerivative
from .basicoperators import SecondDerivative
from .basicoperators import Laplacian
from .basicoperators import Restriction
from .basicoperators import Smoothing1D
from .basicoperators import Smoothing2D

from .avo.poststack import PoststackLinearModelling
from .avo.prestack import PrestackWaveletModelling, PrestackLinearModelling

from .optimization.leastsquares import NormalEquationsInversion, RegularizedInversion
from .optimization.leastsquares import PreconditionedInversion

from .utils.seismicevents import makeaxis, linear2d, parabolic2d
from .utils.tapers import hanningtaper, cosinetaper, taper2d, taper3d
from .utils.wavelets import ricker, gaussian


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'