"""
Basic Linear Operators
======================

The subpackage basicoperators extends some of the basic linear algebra
operations provided by numpy providing forward and adjoint functionalities.

A list of operators present in pylops.basicoperators :

    MatrixMult	                    Matrix multiplication.
    Identity                        Identity operator.
    Zero         	                Zero operator.
    Diagonal     	                Diagonal operator.
    Transpose   	                Transpose operator.
    Flip        	                Flip along an axis.
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
    Real   	                        Real operator.
    Imag   	                        Imag operator.
    Conj   	                        Conj operator.
    Smoothing1D     	            1D Smoothing.
    Smoothing2D	                    2D Smoothing.
    FirstDerivative 	            First derivative.
    SecondDerivative 	            Second derivative.
    Laplacian 	                    Laplacian.
    Gradient 	                    Gradient.
    FirstDirectionalDerivative      First Directional derivative.
    SecondDirectionalDerivative     Second Directional derivative.

"""

from .Block import *
from .BlockDiag import *
from .CausalIntegration import *
from .Conj import *
from .Diagonal import *
from .DirectionalDerivative import *
from .FirstDerivative import *
from .Flip import *
from .FunctionOperator import *
from .Gradient import *
from .HStack import *
from .Identity import *
from .Imag import *
from .Kronecker import *
from .Laplacian import *
from .LinearRegression import *
from .MatrixMult import *
from .MemoizeOperator import *
from .Pad import *
from .Real import *
from .Regression import *
from .Restriction import *
from .Roll import *
from .SecondDerivative import *
from .Smoothing1D import *
from .Smoothing2D import *
from .Spread import *
from .Sum import *
from .Symmetrize import *
from .Transpose import *
from .VStack import *
from .Zero import *

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
