"""
Basic Linear Operators
======================

The subpackage basicoperators extends some of the basic linear algebra
operations provided by numpy providing forward and adjoint functionalities.

A list of operators present in pylops.basicoperators :

    MatrixMult                      Matrix multiplication.
    Identity                        Identity operator.
    Zero                            Zero operator.
    Diagonal                        Diagonal operator.
    Transpose                       Transpose operator.
    Flip                            Flip along an axis.
    Roll                            Roll along an axis.
    Pad                             Pad operator.
    Sum                             Sum operator.
    Symmetrize                      Symmetrize along an axis.
    Restriction                     Restriction (or sampling) operator.
    Regression                      Polynomial regression.
    LinearRegression                Linear regression.
    CausalIntegration               Causal integration.
    Spread                          Spread operator.
    VStack                          Vertical stacking.
    HStack                          Horizontal stacking.
    Block                           Block operator.
    BlockDiag                       Block-diagonal operator.
    Kronecker                       Kronecker operator.
    Real                            Real operator.
    Imag                            Imag operator.
    Conj                            Conj operator.
    Smoothing1D                     1D Smoothing.
    Smoothing2D	                    2D Smoothing.
    FirstDerivative                 First derivative.
    SecondDerivative                Second derivative.
    Laplacian                       Laplacian.
    Gradient                        Gradient.
    FirstDirectionalDerivative      First Directional derivative.
    SecondDirectionalDerivative     Second Directional derivative.

"""

from .functionoperator import *
from .memoizeoperator import *
from .regression import *
from .linearregression import *
from .matrixmult import *
from .diagonal import *
from .zero import *
from .identity import *
from .restriction import *
from .flip import *
from .symmetrize import *
from .spread import *
from .transpose import *
from .roll import *
from .pad import *
from .sum import *
from .vstack import *
from .hstack import *
from .block import *
from .blockdiag import *
from .kronecker import *
from .real import *
from .imag import *
from .conj import *
from .smoothing1d import *
from .smoothing2d import *
from .causalintegration import *
from .firstderivative import *
from .secondderivative import *
from .laplacian import *
from .gradient import *
from .directionalderivative import *


__all__ = [
    "FunctionOperator",
    "MemoizeOperator",
    "Regression",
    "LinearRegression",
    "MatrixMult",
    "Diagonal",
    "Zero",
    "Identity",
    "Restriction",
    "Flip",
    "Symmetrize",
    "Spread",
    "Transpose",
    "Roll",
    "Pad",
    "Sum",
    "VStack",
    "HStack",
    "Block",
    "BlockDiag",
    "Kronecker",
    "Real",
    "Imag",
    "Conj",
    "Smoothing1D",
    "Smoothing2D",
    "CausalIntegration",
    "FirstDerivative",
    "SecondDerivative",
    "Laplacian",
    "Gradient",
    "FirstDirectionalDerivative",
    "SecondDirectionalDerivative",
]
