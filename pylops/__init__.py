"""
PyLops
======

Linear operators and inverse problems are at the core of many of the most used
algorithms in signal processing, image processing, and remote sensing.
When dealing with small-scale problems, the Python numerical scientific
libraries `numpy <http://www.numpy.org>`_
and `scipy <http://www.scipy.org/scipylib/index.html>`_  allow to perform most
of the underlying matrix operations (e.g., computation of matrix-vector
products and manipulation of matrices) in a simple and expressive way.

Many useful operators, however, do not lend themselves to an explicit matrix
representation when used to solve large-scale problems. PyLops operators,
on the other hand, still represent a matrix and can be treated in a similar
way, but do not rely on the explicit creation of a dense (or sparse) matrix
itself. Conversely, the forward and adjoint operators are represented by small
pieces of codes that mimic the effect of the matrix on a vector or
another matrix.

Luckily, many iterative methods (e.g. cg, lsqr) do not need to know the
individual entries of a matrix to solve a linear system. Such solvers only
require the computation of forward and adjoint matrix-vector products as
done for any of the PyLops operators.

PyLops provides
  1. A general construct for creating Linear Operators
  2. An extensive set of commonly used linear operators
  3. A set of least-squares and sparse solvers for linear operators.

Available subpackages
---------------------
basicoperators
    Basic Linear Operators
signalprocessing
    Linear Operators for Signal Processing operations
avo
    Linear Operators for Seismic Reservoir Characterization
waveeqprocessing
    Linear Operators for Wave Equation oriented processing
optimization
    Solvers
utils
    Utility routines

"""

import logging

from .config import *
from .linearoperator import *
from .torchoperator import *
from .pytensoroperator import *
from .jaxoperator import *
from .basicoperators import *
from . import (
    avo,
    basicoperators,
    optimization,
    signalprocessing,
    utils,
    waveeqprocessing,
)
from .avo.poststack import *
from .avo.prestack import *
from .optimization.basic import *
from .optimization.leastsquares import *
from .optimization.sparsity import *
from .utils.seismicevents import *
from .utils.tapers import *
from .utils.utils import *
from .utils.wavelets import *

# Prevent no handler message if an application using PyLops does not configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. pylops should be installed
    # properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")
